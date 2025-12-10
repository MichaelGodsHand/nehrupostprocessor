import os
import json
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
from openai import OpenAI
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logger
logger.add("postprocessor.log", rotation="10 MB", level="INFO")

# FastAPI app
app = FastAPI(title="Nehru Postprocessor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI must be set in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize MongoDB client
def get_mongodb_client():
    """Get MongoDB client connection"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB connection failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")


def normalize_phone_number(phone_number: str) -> str:
    """Normalize phone number to 10 digits"""
    # Remove all non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    # If the number starts with "91" and has more than 10 digits, remove the "91" prefix
    if digits_only.startswith("91") and len(digits_only) > 10:
        phone_number_clean = digits_only[2:]  # Remove first 2 digits (91)
    elif len(digits_only) > 10:
        # If it's longer than 10 digits but doesn't start with 91, take last 10 digits
        phone_number_clean = digits_only[-10:]
    elif len(digits_only) < 10:
        # If less than 10 digits, pad with zeros
        phone_number_clean = digits_only.zfill(10)
    else:
        # Exactly 10 digits
        phone_number_clean = digits_only
    
    return phone_number_clean


def normalize_conversation_tags(conversation: str) -> str:
    """Normalize conversation tags to strict 'User:' and 'Agent:' format"""
    import re
    
    # Split conversation into lines
    lines = conversation.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Pattern to match various agent tag formats (case-insensitive)
        # Matches: "Natalie (Agent):", "Agent (natalie):", "Agent:", "Natalie:", etc.
        agent_pattern = r'^(?:Natalie\s*\(Agent\)|Agent\s*\([^)]*\)|Natalie|Agent)\s*:\s*(.*)$'
        # Pattern to match user tag variations (case-insensitive)
        user_pattern = r'^User\s*:\s*(.*)$'
        
        # Check if line matches agent pattern
        agent_match = re.match(agent_pattern, line, re.IGNORECASE)
        if agent_match:
            # Normalize to "Agent:"
            normalized_lines.append(f"Agent: {agent_match.group(1)}")
        # Check if line matches user pattern
        else:
            user_match = re.match(user_pattern, line, re.IGNORECASE)
            if user_match:
                # Normalize to "User:"
                normalized_lines.append(f"User: {user_match.group(1)}")
            else:
                # Keep line as-is if it doesn't match any pattern
                normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)


def detect_languages(conversation: str) -> List[str]:
    """Detect languages used in the conversation using OpenAI - detects ANY language including transliterated text"""
    try:
        prompt = f"""Analyze the following conversation and identify ALL languages used.
The conversation may contain:
- Multiple languages mixed together (like Tanglish, Hinglish, etc.)
- Languages written in English script/transliteration (e.g., "Vanakkam" is Tamil, "Namaste" is Hindi/Sanskrit)
- Any world language (English, Tamil, Hindi, Malayalam, Kannada, Telugu, Spanish, French, Arabic, etc.)

IMPORTANT:
- Detect languages even when words are transliterated in English letters
- Include ALL languages present, not just a limited set
- Return language names in lowercase English (e.g., "tamil", "hindi", "spanish", "french")
- If only English is used, return ["english"]
- If multiple languages are mixed, include all of them

Return the result as a JSON object with a "languages" field containing an array:
{{"languages": ["english", "tamil"]}}

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert language detection assistant. You can detect ANY language in conversations, including transliterated text (words written in English script but belonging to other languages). Return a JSON object with a 'languages' array containing all detected languages in lowercase."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract languages array from response
        languages = result.get("languages", [])
        
        # Normalize language names to lowercase and remove empty strings
        languages = [lang.lower().strip() for lang in languages if lang and lang.strip()]
        
        # If no languages detected, default to english
        if not languages:
            languages = ["english"]
        
        # Remove duplicates and sort
        languages = sorted(list(set(languages)))
        
        logger.info(f"Detected languages: {languages}")
        return languages
        
    except Exception as e:
        logger.error(f"Error detecting languages: {e}")
        # Default to english if detection fails
        return ["english"]


def detect_scholarship_interests(conversation: str) -> List[str]:
    """Detect courses the caller is interested in for scholarships using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and identify ALL courses/programs that the caller has expressed interest in for scholarships.

Look for:
- Courses mentioned when discussing scholarships (e.g., "I want scholarship for Computer Science", "scholarship for AIML")
- Programs the caller is interested in pursuing (e.g., "I'm interested in Computer Science and also AIML")
- Any course names mentioned in the context of scholarship inquiries
- Course abbreviations (e.g., "CSE", "AIML", "ECE", "ME", "IT", "EEE")
- Full course names (e.g., "Computer Science and Engineering", "Artificial Intelligence and Machine Learning")

IMPORTANT:
- Extract course names as mentioned in the conversation (preserve original casing if significant, but normalize common variations)
- Include ALL courses mentioned, not just one
- If no scholarship-related course interests are found, return an empty array []
- Course names can be in any format (full names, abbreviations, etc.) - extract them as mentioned
- Common course names at Nehru College include: Computer Science and Engineering, Information Technology, Electronics and Communication Engineering, Electrical and Electronics Engineering, Mechanical Engineering, AIML, etc.

Return the result as a JSON object with a "scholarships" field containing an array:
{{"scholarships": ["computer science and engineering", "AIML"]}}

If no courses are mentioned in scholarship context, return:
{{"scholarships": []}}

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a course interest extraction assistant. Extract all courses/programs that the caller has expressed interest in for scholarships. Return a JSON object with a 'scholarships' array containing course names as mentioned in the conversation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract scholarships array from response
        scholarships = result.get("scholarships", [])
        
        # Normalize and clean course names - remove empty strings
        scholarships = [course.strip() for course in scholarships if course and course.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_scholarships = []
        for course in scholarships:
            course_lower = course.lower()
            if course_lower not in seen:
                seen.add(course_lower)
                unique_scholarships.append(course)
        
        logger.info(f"Detected scholarship interests: {unique_scholarships}")
        return unique_scholarships
        
    except Exception as e:
        logger.error(f"Error detecting scholarship interests: {e}")
        # Return empty array if detection fails
        return []


def extract_phone_number(conversation: str) -> Optional[str]:
    """Extract phone number from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract the phone number mentioned in it.
The phone number could be in various formats (10 digits, with country code, with spaces, etc.).
Return ONLY the phone number in digits (no spaces, no dashes, no plus signs).
If no phone number is found, return "NOT_FOUND".

Conversation:
{conversation}

Phone number:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a phone number extraction assistant. Extract phone numbers from conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        
        phone_number = response.choices[0].message.content.strip()
        
        if phone_number == "NOT_FOUND" or not phone_number:
            logger.warning("No phone number found in conversation")
            return None
        
        # Normalize the extracted phone number
        normalized = normalize_phone_number(phone_number)
        
        if len(normalized) != 10:
            logger.warning(f"Invalid phone number format extracted: {phone_number}")
            return None
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error extracting phone number: {e}")
        return None


def extract_email(conversation: str) -> Optional[str]:
    """Extract email address from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract the email address mentioned in it.
The email address should be in standard format (e.g., user@example.com).
Return ONLY the email address in lowercase.
If no email address is found, return "NOT_FOUND".

Conversation:
{conversation}

Email address:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an email extraction assistant. Extract email addresses from conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        email = response.choices[0].message.content.strip()
        
        if email == "NOT_FOUND" or not email:
            logger.warning("No email address found in conversation")
            return None
        
        # Normalize email (lowercase, trim)
        email = email.lower().strip()
        
        # Basic email validation
        if "@" not in email or "." not in email.split("@")[1]:
            logger.warning(f"Invalid email format extracted: {email}")
            return None
        
        return email
        
    except Exception as e:
        logger.error(f"Error extracting email: {e}")
        return None


def create_user_from_conversation(conversation: str, phone_number: Optional[str] = None, email: Optional[str] = None) -> Optional[dict]:
    """Create user record from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract the user's information.
Extract the following information:
1. Name: The person's full name
2. Email: The person's email address

Return the information in JSON format:
{{
    "name": "Full Name",
    "email": "email@example.com"
}}

If any information is not found, use empty string "" for that field.

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Extract user information from conversations and return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        user_data = json.loads(response.choices[0].message.content)
        
        # Use extracted email from function parameter if available, otherwise use extracted from conversation
        extracted_email = email or user_data.get("email", "")
        if extracted_email:
            extracted_email = extracted_email.lower().strip()
        
        # Create user document
        user_doc = {
            "name": user_data.get("name", ""),
            "email": extracted_email
        }
        
        # Add phone number if available
        if phone_number:
            phone_db_format = f"91{phone_number}"
            user_doc["phone_number"] = phone_db_format
        
        # Insert into MongoDB
        client = get_mongodb_client()
        db = client["NEHRU"]
        users_collection = db["users"]
        
        # Check if user already exists by phone number (if provided)
        if phone_number:
            phone_db_format = f"91{phone_number}"
            existing = users_collection.find_one({"phone_number": phone_db_format})
            if existing:
                logger.info(f"User already exists with phone: {phone_db_format}")
                client.close()
                return existing
        
        # Check if user already exists by email (if provided)
        if extracted_email:
            existing = users_collection.find_one({"email": extracted_email})
            if existing:
                logger.info(f"User already exists with email: {extracted_email}")
                client.close()
                return existing
        
        # Insert new user
        result = users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        
        identifier = f"phone: {phone_db_format}" if phone_number else f"email: {extracted_email}"
        logger.info(f"Created new user: {user_doc['name']} ({identifier})")
        client.close()
        
        return user_doc
        
    except Exception as e:
        logger.error(f"Error creating user from conversation: {e}")
        return None


def format_budget_indian_style(budget_text: str) -> str:
    """Format budget in Indian number format (e.g., 1,90,000-2,00,000)"""
    try:
        # Handle None or empty values
        if not budget_text or not isinstance(budget_text, str):
            return ""
        
        # If already in correct format, return as is
        if "-" in budget_text and "," in budget_text and "lakh" not in budget_text.lower() and "per" not in budget_text.lower():
            return budget_text.strip()
        
        prompt = f"""Convert the following budget information to Indian number format with commas.
CRITICAL FORMAT REQUIREMENT: The budget MUST ALWAYS follow this exact format: X,XX,XXX-Y,YY,YYY (e.g., 1,90,000-2,00,000)

Rules:
- Always assume the budget is for YEARLY payment (even if mentioned as monthly, convert to yearly by multiplying by 12)
- Format: Use Indian numbering system with commas - ALWAYS use format: X,XX,XXX-Y,YY,YYY
- The format MUST be: [lower_amount with commas]-[upper_amount with commas]
- If a range is given (e.g., "1.5 to 2 Lakhs"), convert to: 1,50,000-2,00,000
- If single amount (e.g., "2 Lakhs"), convert to range format: 2,00,000-2,00,000
- Remove words like "per year", "per month", "lakhs", "lakh", "rupees", "rs", etc.
- ALWAYS output in format: X,XX,XXX-Y,YY,YYY (with commas in Indian style) - THIS IS MANDATORY
- If monthly amount is mentioned, multiply by 12 to get yearly
- The format convention is STRICT: always use commas in Indian numbering style (e.g., 1,90,000-2,00,000)

Examples (STRICT FORMAT):
- "1.5 - 2 Lakhs per year" → "1,50,000-2,00,000"
- "1.9 to 2 Lakhs" → "1,90,000-2,00,000"
- "35,000 per month" → "4,20,000-4,20,000" (35,000 * 12 = 4,20,000)
- "2-3 Lakhs" → "2,00,000-3,00,000"
- "50,000 per month" → "6,00,000-6,00,000"

REMEMBER: The output format MUST ALWAYS be: X,XX,XXX-Y,YY,YYY (e.g., 1,90,000-2,00,000) - NO EXCEPTIONS

Budget text: {budget_text}

Return ONLY the formatted budget in Indian number format following the strict format: X,XX,XXX-Y,YY,YYY (e.g., "1,90,000-2,00,000"):"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a budget formatting assistant. Convert budget amounts to Indian number format with commas, always assuming yearly payment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        formatted_budget = response.choices[0].message.content.strip()
        # Remove any quotes if present
        formatted_budget = formatted_budget.strip('"').strip("'")
        
        logger.info(f"Formatted budget: {budget_text} → {formatted_budget}")
        return formatted_budget
        
    except Exception as e:
        logger.error(f"Error formatting budget: {e}, using original: {budget_text}")
        return budget_text


def detect_follow_up(conversation: str) -> bool:
    """Detect if the caller agreed to a follow-up in the conversation"""
    try:
        prompt = f"""Analyze the following conversation and determine if the caller has agreed to a follow-up call or meeting.

Look for:
- Explicit agreement to follow-up (e.g., "yes, call me back", "I'll be available", "sure, follow up")
- Agreement to schedule a call or meeting later
- Positive responses to follow-up requests
- Expressions like "yes", "sure", "okay", "alright" in response to follow-up questions

Return true ONLY if there is clear agreement to a follow-up. If uncertain or no agreement, return false.

Return the result as a JSON object:
{{"follow_up": true}} or {{"follow_up": false}}

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a follow-up detection assistant. Determine if the caller agreed to a follow-up call or meeting. Return a JSON object with a boolean 'follow_up' field."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        follow_up = result.get("follow_up", False)
        
        # Ensure it's a boolean
        if isinstance(follow_up, str):
            follow_up = follow_up.lower() in ["true", "yes", "1"]
        
        logger.info(f"Detected follow_up: {follow_up}")
        return bool(follow_up)
        
    except Exception as e:
        logger.error(f"Error detecting follow-up: {e}")
        return False


def generate_analytics(conversation: str, user_id: ObjectId) -> Optional[dict]:
    """Generate analytics from conversation using OpenAI"""
    try:
        prompt = f"""Analyze the following conversation and extract analytics information about the user.
Extract the following information:
1. course_interest: The engineering branch or course they're interested in (e.g., "computer science engineering", "mechanical engineering", etc.)
2. city: The city they're from or located in
3. budget: Their budget range for education (can be in any format - lakhs, per month, per year, etc. - we'll format it later)
4. hostel_needed: Boolean value (true or false) - whether they need hostel accommodation
5. intent_level: One of "TOFU" (Top of Funnel - early interest), "MOFU" (Middle of Funnel - considering), or "BOFU" (Bottom of Funnel - ready to enroll)

Return the information in JSON format:
{{
    "course_interest": "course name in lowercase",
    "city": "city name",
    "budget": "budget range as mentioned (any format)",
    "hostel_needed": true or false,
    "intent_level": "TOFU" or "MOFU" or "BOFU"
}}

Conversation:
{conversation}

JSON:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an analytics extraction assistant. Extract user analytics from conversations and return valid JSON with the exact fields specified."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        analytics_data = json.loads(response.choices[0].message.content)
        
        # Ensure hostel_needed is boolean
        hostel_needed = analytics_data.get("hostel_needed", False)
        if isinstance(hostel_needed, str):
            hostel_needed = hostel_needed.lower() in ["true", "yes", "1", "needed", "required"]
        
        # Format budget in Indian number format - STRICT FORMAT: X,XX,XXX-Y,YY,YYY (e.g., 1,90,000-2,00,000)
        budget_raw = analytics_data.get("budget", "")
        budget_formatted = format_budget_indian_style(budget_raw) if budget_raw else ""
        
        # Detect follow-up
        follow_up = detect_follow_up(conversation)
        
        # Create analytics document
        # Handle None values safely - get() returns None if key exists with None value
        course_interest = analytics_data.get("course_interest") or ""
        course_interest = course_interest.lower() if isinstance(course_interest, str) else ""
        
        city = analytics_data.get("city") or ""
        
        intent_level = analytics_data.get("intent_level") or "TOFU"
        intent_level = intent_level.upper() if isinstance(intent_level, str) else "TOFU"
        
        analytics_doc = {
            "user_id": user_id,
            "course_interest": course_interest,
            "city": city,
            "budget": budget_formatted,
            "hostel_needed": bool(hostel_needed),
            "intent_level": intent_level,
            "follow_up": follow_up
        }
        
        return analytics_doc
        
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        return None


# Pydantic models
class ConversationRequest(BaseModel):
    conversation: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "postprocessor"}


@app.post("/process")
async def process_conversation(request: ConversationRequest):
    """
    Process conversation data:
    Phase 1: Extract phone number, check/create user
    Phase 2: Generate/update analytics
    """
    try:
        conversation = request.conversation
        
        if not conversation or not conversation.strip():
            raise HTTPException(status_code=400, detail="Conversation data is required")
        
        logger.info("Starting conversation processing...")
        
        # Phase 1: Extract phone number or email, detect languages, and check/create user
        logger.info("Phase 1: Extracting phone number and detecting languages...")
        phone_number = extract_phone_number(conversation)
        email = None
        
        # If phone number is not found, try to extract email
        if not phone_number:
            logger.info("No phone number found, attempting to extract email...")
            email = extract_email(conversation)
            
            if not email:
                raise HTTPException(status_code=400, detail="Could not extract phone number or email from conversation")
            
            logger.info(f"Extracted email: {email}")
        else:
            logger.info(f"Extracted phone number: {phone_number}")
        
        # Detect languages used in the conversation
        logger.info("Phase 1: Detecting languages used in conversation...")
        languages_used = detect_languages(conversation)
        logger.info(f"Detected languages: {languages_used}")
        
        # Detect scholarship interests
        logger.info("Phase 1: Detecting scholarship interests...")
        scholarships = detect_scholarship_interests(conversation)
        logger.info(f"Detected scholarship interests: {scholarships}")
        
        # Check if user exists
        client = get_mongodb_client()
        db = client["NEHRU"]
        users_collection = db["users"]
        analytics_collection = db["userAnalytics"]
        
        user = None
        
        # Try to find user by phone number first (if available)
        if phone_number:
            phone_db_format = f"91{phone_number}"
            user = users_collection.find_one({"phone_number": phone_db_format})
        
        # If not found by phone, try to find by email (if available)
        if not user and email:
            email_normalized = email.lower().strip()
            user = users_collection.find_one({"email": email_normalized})
        
        if not user:
            # User doesn't exist - create new user
            logger.info("User not found, creating new user...")
            user = create_user_from_conversation(conversation, phone_number=phone_number, email=email)
            
            if not user:
                client.close()
                raise HTTPException(status_code=500, detail="Failed to create user")
        else:
            logger.info(f"User found: {user.get('name', 'Unknown')}")
        
        user_id = user.get("_id")
        
        if not user_id:
            client.close()
            raise HTTPException(status_code=500, detail="User ID not found")
        
        # Phase 2: Generate/update analytics
        logger.info("Phase 2: Generating analytics...")
        analytics_doc = generate_analytics(conversation, user_id)
        
        if not analytics_doc:
            client.close()
            raise HTTPException(status_code=500, detail="Failed to generate analytics")
        
        # Check if analytics already exists for this user
        existing_analytics = analytics_collection.find_one({"user_id": ObjectId(user_id)})
        
        if existing_analytics:
            # Update existing analytics
            logger.info("Updating existing analytics...")
            
            # If follow_up field is missing in existing analytics, add it
            if "follow_up" not in existing_analytics:
                logger.info("follow_up field missing in existing analytics, adding it...")
                # Detect follow-up from current conversation
                follow_up = detect_follow_up(conversation)
                analytics_doc["follow_up"] = follow_up
                logger.info(f"Added follow_up field: {follow_up}")
            
            # Update with new analytics data
            analytics_collection.update_one(
                {"user_id": ObjectId(user_id)},
                {"$set": analytics_doc}
            )
            analytics_doc["_id"] = existing_analytics.get("_id")
            logger.info("Analytics updated successfully")
        else:
            # Insert new analytics
            logger.info("Creating new analytics...")
            result = analytics_collection.insert_one(analytics_doc)
            analytics_doc["_id"] = result.inserted_id
            logger.info("Analytics created successfully")
        
        # Phase 3: Store conversation history
        logger.info("Phase 3: Storing conversation history...")
        conversation_history_collection = db["conversationHistory"]
        
        # Normalize conversation tags to strict "User:" and "Agent:" format
        normalized_conversation = normalize_conversation_tags(conversation)
        logger.info("Normalized conversation tags to strict 'User:' and 'Agent:' format")
        
        # Always create a new conversation history document (users can have many conversations)
        conversation_history_doc = {
            "user_id": ObjectId(user_id),
            "conversation": normalized_conversation,  # Store the normalized conversation with strict tags
            "timestamp": datetime.now(),  # Add timestamp to track when conversation occurred
            "languages_used": languages_used,  # Add languages detected in Phase 1
            "scholarships": scholarships  # Add scholarship interests detected in Phase 1
        }
        
        # Insert new conversation history (always create new document)
        logger.info("Creating new conversation history document...")
        result = conversation_history_collection.insert_one(conversation_history_doc)
        conversation_history_doc["_id"] = result.inserted_id
        logger.info(f"Conversation history created successfully with ID: {result.inserted_id}")
        
        client.close()
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Conversation processed successfully",
            "user": {
                "_id": str(user.get("_id")),
                "name": user.get("name"),
                "email": user.get("email"),
                "phone_number": user.get("phone_number")
            },
            "analytics": {
                "_id": str(analytics_doc.get("_id")),
                "user_id": str(analytics_doc.get("user_id")),
                "course_interest": analytics_doc.get("course_interest"),
                "city": analytics_doc.get("city"),
                "budget": analytics_doc.get("budget"),
                "hostel_needed": analytics_doc.get("hostel_needed"),
                "intent_level": analytics_doc.get("intent_level")
            },
            "conversation_history": {
                "_id": str(conversation_history_doc.get("_id")),
                "user_id": str(conversation_history_doc.get("user_id")),
                "conversation": conversation_history_doc.get("conversation")
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
