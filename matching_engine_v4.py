# matching_engine_v4.py

"""
Business Activity Matching Engine V4
Enhanced: Multi-Profile Intent Detection + Expert Reranking
Final Logic for Multi-Activity and Close-Match Visibility.
"""

import pandas as pd
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient
import json
from typing import List, Dict
import re
from fuzzywuzzy import fuzz
import os
import time
import numpy as np
from datetime import datetime

# --- Configuration (Copied from V3) ---
SYNONYM_FILE = "synonym_dictionary.json"
CACHE_FILE = "query_cache.json"
SIMILARITY_THRESHOLD = 0.95 
CSV_FILE = "enhanced_activities_manual.csv"
COLLECTION_NAME = "business_activities"

# Scoring weights (Your existing V3 weights, used to calculate 'final_score')
WEIGHTS = {
    'semantic': 0.35,
    'keyword': 0.65,
    'phrase_match_bonus': 25,
    'category_boost': 20
}

# FINAL ULTIMATE SCORE Weights (The new expert validation score)
ULTIMATE_WEIGHTS = {
    'llm_match_score': 0.90,
    'hybrid_score': 0.10,
}
# New penalty threshold for surgical demotion
LLM_PENALTY_THRESHOLD = 50 
LLM_PENALTY_MULTIPLIER = 0.01

# Reranking Strategy Configuration
USE_TWO_STAGE_RERANKING = False  # Set to False to revert to original single-stage reranking

# Two-Stage Reranking Parameters (only used if USE_TWO_STAGE_RERANKING = True)
QUICK_FILTER_POOL = 100  # Stage 1: Quick filter evaluates this many
DEEP_ANALYSIS_POOL = 30  # Stage 2: Deep analysis evaluates this many (top from Stage 1)

# --- Credential Handling ---
try:
    # Attempt to load Streamlit secrets
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
    QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
except:
    # Load from environment variables (for local run)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Setup for Embedding and LLM models
Settings.embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
# Use a fast model for basic tasks and a more powerful one for reranking
Settings.llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini") 


class ActivityMatcherV4:
    def __init__(self):
        """Initialize the matching engine with Qdrant Cloud"""
        print("🔧 Loading matching engine V4 (Cloud)...")
        
        # Connect to Qdrant Cloud
        print("   Connecting to Qdrant Cloud...")
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        # Load synonyms
        print("   Loading synonym dictionary...")
        try:
            with open(SYNONYM_FILE, 'r') as f:
                self.synonyms = json.load(f)
        except FileNotFoundError:
            print(f"WARNING: {SYNONYM_FILE} not found. Synonym expansion disabled.")
            self.synonyms = {}

        # Load activities CSV
        print("   Loading activities data...")
        self.df = pd.read_csv(CSV_FILE, encoding='latin-1')
        
        print("✅ Matching engine V4 ready!\n")
    

    def _load_cache(self) -> Dict:
        """Load query cache from file"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {"queries": []}
        return {"queries": []}
    
    def _save_cache(self, cache: Dict):
        """Save query cache to file"""
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query using the same model as semantic search"""
        embedding = Settings.embed_model.get_text_embedding(query)
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _check_cache(self, query: str, query_embedding: List[float]) -> tuple:
        """
        Check if similar query exists in cache
        Returns: (cache_hit: bool, cached_result: Dict or None, similarity: float)
        """
        cache = self._load_cache()
        
        best_match = None
        best_similarity = 0.0
        
        for cached_query in cache.get("queries", []):
            cached_embedding = cached_query.get("embedding", [])
            if not cached_embedding:
                continue
            
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_query
        
        # Cache hit if similarity is above threshold
        if best_similarity >= SIMILARITY_THRESHOLD and best_match:
            print(f"✅ CACHE HIT! Similarity: {best_similarity:.2%} with query: '{best_match['query']}'")
            return (True, best_match["result"], best_similarity)
        
        return (False, None, best_similarity)
    
    def _add_to_cache(self, query: str, query_embedding: List[float], result: Dict):
        """Add query and result to cache"""
        cache = self._load_cache()
        
        # Add new entry
        cache_entry = {
            "query": query,
            "embedding": query_embedding,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        if "queries" not in cache:
            cache["queries"] = []
        
        cache["queries"].append(cache_entry)
        
        # Keep only last 100 queries to prevent file from growing too large
        if len(cache["queries"]) > 100:
            cache["queries"] = cache["queries"][-100:]
        
        self._save_cache(cache)
        print(f"💾 Cached query: '{query}'")
    # --- HELPER FUNCTIONS COPIED FROM V3 ---
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms (Copied from V3)"""
        expanded_terms = [query]
        words = query.lower().split()
        
        for word in words:
            for key, synonyms_list in self.synonyms.items():
                if word == key or word in synonyms_list:
                    expanded_terms.extend(synonyms_list)
                    expanded_terms.append(key)
        
        expanded = ' '.join(list(set(expanded_terms)))
        return expanded
    
    def detect_category(self, query: str) -> str:
        """Detect likely category from query (Copied from V3)"""
        query_lower = query.lower()
        
        category_keywords = {
            'ICT': ['software', 'website', 'app', 'technology', 'IT', 'cloud', 'digital', 'cyber', 'coding', 'programming', 'data', 'internet', 'online platform', 'web', 'tech', 'computer', 'system', 'network', 'development'],
            'F&B': ['restaurant', 'cafe', 'food', 'catering', 'kitchen', 'dining', 'bakery', 'eatery', 'coffee shop', 'bar', 'pub', 'bistro', 'canteen', 'cafeteria', 'food service'],
            'Rentals': ['rental', 'rent', 'lease', 'leasing', 'hiring', 'letting', 'accommodation', 'property rental', 'equipment rental', 'vehicle rental', 'space rental'],
            'Trading': ['import', 'export', 'wholesale', 'trading', 'goods', 'retail', 'shop', 'store', 'sell', 'sell online', 'digital sales', 'e-commerce', 'online selling', 'commodity', 'dealer', 'supplier', 'distributor', 'purchase'],
            'Professional': ['consulting', 'advisory', 'legal', 'accounting', 'marketing', 'consultancy', 'lawyer', 'management', 'audit', 'tax', 'strategy', 'business services', 'professional services', 'bookkeeping', 'financial advice', 'corporate', 'attorney'],
            'Manufacturing': ['production', 'manufacturing', 'factory', 'fabrication', 'making', 'assembly', 'processing', 'industrial production', 'textile manufacturing', 'garment production', 'metal works'],
            'Healthcare': ['medical', 'clinic', 'healthcare', 'dental', 'hospital', 'pharmacy', 'doctor', 'health', 'medicine', 'patient care', 'nursing', 'physiotherapy', 'laboratory', 'diagnostic', 'surgery', 'treatment'],
            'Transportation': ['transport', 'logistics', 'shipping', 'cargo', 'delivery', 'courier', 'freight', 'moving', 'haulage', 'trucking', 'vehicle', 'taxi', 'bus', 'marine transport', 'air cargo', 'warehousing', 'tour packages', 'tour operator'],
            'Education': ['training', 'education', 'school', 'courses', 'institute', 'learning', 'teaching', 'coaching', 'fitness coaching', 'tutoring', 'academy', 'university', 'college', 'kindergarten', 'nursery', 'workshop'],
            'Administrative': ['administration', 'administrative support', 'office management', 'secretarial', 'business support', 'document preparation', 'data entry', 'filing', 'clerical', 'tour operator', 'travel agency', 'event management'],
            'Financial': ['finance', 'banking', 'investment', 'insurance', 'loan', 'credit', 'financial services', 'money exchange', 'forex', 'stock', 'securities', 'fund management', 'wealth management', 'mortgage', 'debt', 'invoice purchase', 'factoring'],
            'Agriculture': ['agriculture', 'farming', 'crop', 'livestock', 'agricultural', 'cultivation', 'plantation', 'animal breeding', 'poultry', 'dairy', 'fishery', 'horticulture'],
            'Mining': ['mining', 'quarrying', 'extraction', 'oil', 'gas', 'petroleum', 'coal', 'minerals', 'ore', 'drilling', 'excavation'],
            'Construction': ['construction', 'building', 'contractor', 'civil engineering', 'renovation', 'demolition', 'plumbing', 'electrical work', 'carpentry', 'masonry', 'project development'],
            'Art': ['art', 'design', 'creative', 'graphic', 'photography', 'painting', 'sculpture', 'gallery', 'artistic', 'illustration', 'visual arts', 'craft', 'media production', 'animation', 'entertainment'],
            'Services': ['cleaning', 'security', 'maintenance', 'repair', 'service', 'fix', 'lifestyle coaching', 'personal services', 'beauty', 'salon', 'spa', 'laundry', 'pest control', 'facility management']
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return category
        return None
    
    def keyword_score(self, query: str, keywords: str, activity_name: str) -> float:
        """Calculate keyword matching score (Copied from V3)"""
        if not keywords or keywords in ['-', 'ERROR']:
            return 0
        
        query_lower = query.lower()
        keywords_lower = keywords.lower()
        activity_lower = activity_name.lower()
        
        score = 0
        
        # Exact phrase match in keywords
        if query_lower in keywords_lower:
            score += 100
        elif query_lower in activity_lower:
            score += 90
        else:
            # Word-level matching
            query_words = set(query_lower.split())
            keyword_words = set(keywords_lower.replace(',', ' ').split())
            activity_words = set(activity_lower.split())
            
            keyword_matches = len(query_words & keyword_words)
            activity_matches = len(query_words & activity_words)
            
            if len(query_words) > 0:
                keyword_match_ratio = keyword_matches / len(query_words)
                activity_match_ratio = activity_matches / len(query_words)
                score = (keyword_match_ratio * 70) + (activity_match_ratio * 30)
            
            # Fuzzy matching
            fuzzy_keyword = fuzz.partial_ratio(query_lower, keywords_lower)
            fuzzy_activity = fuzz.partial_ratio(query_lower, activity_lower)
            fuzzy_score = max(fuzzy_keyword, fuzzy_activity) * 0.3
            score += fuzzy_score
        
        return min(score, 100)

    def detect_multiple_intents(self, query: str) -> List[Dict]:
        """Detect multiple business intents in query (Copied from V3, used internally by V3 scoring logic)"""
        query_lower = query.lower()
        intents = []
        
        # Intent patterns
        intent_patterns = {
            'e-commerce': {
                'keywords': ['sell online', 'online shop', 'e-commerce', 'website selling', 'internet sales', 'online store', 'digital sales', 'online selling'],
                'activities': ['4790.00', '4791.00']
            },
            'retail_physical': {
                'keywords': ['retail shop', 'physical store', 'brick and mortar', 'shop', 'retail store', 'outlet'],
                'category_filter': 'Trading'
            },
            'import_export': {
                'keywords': ['import', 'export', 'international trade', 'wholesale'],
                'category_filter': 'Trading'
            },
            'consulting': {
                'keywords': ['consulting', 'advisory', 'consultant', 'advice'],
                'category_filter': 'Professional'
            },
            'content_creation': {
                'keywords': ['content creation', 'youtube', 'social media content', 'video content', 'digital content'],
                'activities': ['6202.11', '6311.95']
            },
            'tour_travel': {
                'keywords': ['tour packages', 'travel agency', 'tour operator', 'tourism'],
                'activities': ['7912.00']
            }
        }
        
        # Check for each intent
        for intent_name, intent_data in intent_patterns.items():
            for keyword in intent_data['keywords']:
                if keyword in query_lower:
                    intents.append({
                        'name': intent_name,
                        'keyword': keyword,
                        'activities': intent_data.get('activities', []),
                        'category': intent_data.get('category_filter', None)
                    })
                    break
        
        return intents
    
    # --- NEW V4 LOGIC: Multi-Profile Intent Extraction ---
    def _get_intent_and_profiles(self, query: str) -> Dict:
        """Uses LLM to extract primary and secondary activity profiles."""
        
        intent_llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini") 
        
        valid_categories = "[Trading, Professional, Financial, Services, Manufacturing, Agriculture, Other]"
        
        prompt = f"""
        You are an expert Meydan Free Zone Licensing Consultant. Analyze the customer query and extract up to TWO distinct activity profiles, as a single business may require multiple licenses (e.g., Trading AND E-commerce).
        
        **Customer Query:** {query}

        **Instructions:**
        1.  **Determine Primary Profile:** The core business function (highest revenue or risk).
        2.  **Determine Secondary Profile (if applicable):** A distinct supporting or secondary function (e.g., online sales, consultancy). If only one is needed, use the same data for both profiles.
        3.  **For EACH profile,** select a single best Category Hint from the list: {valid_categories} and extract 3-5 specific, professional business intents/keywords.

        **Output ONLY a JSON object** with the following two keys: "primary_profile" and "secondary_profile".
        """
        try:
            response = intent_llm.complete(prompt)
            # The LLM may return non-JSON text; strip it aggressively
            json_text = response.text.strip()
            
            # Use regex to find and extract the JSON object if the LLM wrapped it in markdown
            match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                # If no JSON is found, raise an error to trigger the fallback gracefully
                raise ValueError("LLM did not return a valid JSON structure.")

        except Exception as e:
            print(f"FATAL: Intent extraction failed: {e}. Falling back to generic query terms.")
            # THIS FALLBACK IS WHAT YOU ARE SEEING:
            return {
                "primary_profile": {"category_hint": "Services", "intents": [query]},
                "secondary_profile": {"category_hint": "Other", "intents": []}
            }

    # --- NEW: Stage 1 Quick Filter (Fast pre-filtering with GPT-4o-mini) ---
    def _quick_filter_with_mini(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Uses GPT-4o-mini for fast initial scoring of all candidates."""
        
        mini_llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        
        if not candidates:
            return []
        
        # Create a condensed list for quick evaluation (code + name only, no full descriptions)
        candidates_text = ""
        for idx, activity in enumerate(candidates):
            code = activity.get('code', 'N/A')
            name = activity.get('activity_name', 'N/A')
            candidates_text += f"{idx + 1}. Code {code}: {name}\n"
        
        quick_filter_prompt = f"""
        You are a licensing expert. Quickly score how relevant each activity is to the customer's query.
        
        **Customer Query:** {query}
        
        **Activities (Code and Name only):**
        {candidates_text}
        
        **Instructions:**
        - Score each activity 0-100 based on relevance
        - Output ONLY a JSON array with format: [{{"Code": "1234.00", "score": 85}}, ...]
        - Be generous - we'll do detailed analysis later
        """
        
        try:
            response = mini_llm.complete(quick_filter_prompt)
            json_text = response.text.strip()
            
            # Extract JSON array
            match = re.search(r'\[\s*\{.*\}\s*\]', json_text, re.DOTALL)
            if match:
                scores_array = json.loads(match.group(0))
            else:
                scores_array = json.loads(json_text)
            
            # Map scores back to candidates
            score_lookup = {str(item.get('Code')): item.get('score', 0) for item in scores_array}
            
            for activity in candidates:
                code_str = str(activity['code'])
                activity['quick_filter_score'] = score_lookup.get(code_str, 0)
            
            # Sort by quick filter score
            candidates.sort(key=lambda x: x.get('quick_filter_score', 0), reverse=True)
            
            return candidates
            
        except Exception as e:
            print(f"WARNING: Quick filter failed: {e}. Using original order.")
            # Return candidates in original order if quick filter fails
            return candidates

    # --- NEW V4 LOGIC: Expert Reranking Layer (Core of the V4 upgrade) ---
    def _rerank_with_llm(self, query: str, top_candidates: List[Dict]) -> List[Dict]:
        """Uses a high-capacity LLM (GPT-4) to apply expert reasoning and score all top candidates (0-100)."""
        
        # Use a high-capacity model for critical reranking (Ensure API Key supports this model)
        reranker_llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o") 

        if not top_candidates:
            return []

        candidates_text = ""
        for idx, activity in enumerate(top_candidates):
            name = activity.get('activity_name', 'N/A')
            desc = activity.get('description', 'No Description')
            code = activity.get('code', 'N/A')
            keywords = activity.get('keywords', 'None')
            
            candidates_text += f"\n--- Candidate {idx + 1} (Code: {code}) ---\n"
            candidates_text += f"Activity Name: {name}\n"
            candidates_text += f"Description: {desc}\n"
            candidates_text += f"Keywords: {keywords}\n"
            candidates_text += "---------------------------------------\n"

        rerank_prompt = f"""
        You are the Chief Licensing Expert for the Meydan Free Zone. Critically evaluate the relevance of each activity against the customer's detailed query, which may require multiple activities.

        **Customer's Business Query:** {query}

        **List of Activities to Evaluate:**
        {candidates_text}

        **Instructions:**
        1.  **Score All Candidates:** For EVERY activity in the list, assign an "llm_match_score" (0-100) based on how well it fits ANY part of the customer's query.
       

        **Output ONLY a single JSON array of objects**, one for each candidate, containing ONLY the following keys: Code and llm_match_score.

        """
        
        try:
            response = reranker_llm.complete(rerank_prompt) 
            json_text = response.text.strip()
            
            # --- CRITICAL FIX: Robustly extract JSON array from LLM response ---
            # Search for the outermost JSON array structure [ ... ]
            match = re.search(r'\[\s*\{.*\}\s*\]', json_text, re.DOTALL)
            
            if match:
                # Parse the isolated JSON text
                reranking_array = json.loads(match.group(0))
            else:
                # Fallback: Try parsing the raw text, but if it fails, trigger the exception
                reranking_array = json.loads(json_text)
            
            # --- End CRITICAL FIX ---
            
            # Map LLM results to the candidate list
            llm_data_lookup = {}
            for item in reranking_array:
                # Use 'Code' key from the LLM output (uppercase) for lookup
                llm_data_lookup[str(item.get('Code'))] = item
                # Also handle conversion to integer/float just in case the key is missing from the LLM output
                try:
                    llm_data_lookup[str(int(float(item.get('Code'))))] = item
                except:
                    pass
            
            # Merge the new data back into the original candidates list
            for activity in top_candidates:
                code_str = str(activity['code']) 
                llm_data = llm_data_lookup.get(code_str, {})
                
                activity['llm_match_score'] = llm_data.get('llm_match_score', 0)
                activity['llm_rationale'] = llm_data.get('llm_rationale', 'Expert Rationale generated.') # Default rationale for success
                
            return top_candidates
            
        except Exception as e:
            # THIS IS THE FALLBACK YOU ARE CURRENTLY SEEING
            print(f"FATAL: LLM Reranking failed due to parsing error: {e}. Returning scores of 0.")
            for activity in top_candidates:
                activity['llm_match_score'] = 0.0
                activity['llm_rationale'] = "Reranking failed due to LLM communication/parsing error."
            return top_candidates

    # --- V3 SCORING LOGIC WRAPPED IN HELPER FUNCTION ---
    def _run_hybrid_search_and_score(self, query: str, all_intents: List[str]) -> List[Dict]:
        """Contains the exact V3 scoring logic to calculate 'final_score' and returns the full list."""
        
        print(f"🔍 V3 Hybrid Scoring Logic running for: '{query}'")
        
        # Detect multiple intents (V3 logic is dependent on this function's output)
        intents = self.detect_multiple_intents(query)
        
        # Expand query
        expanded_query = self.expand_query(query)
        
        # Detect category
        category_hint = self.detect_category(query)
        
        # Semantic search
        # Using a larger retrieval pool to ensure we capture relevant activities
        retriever = self.index.as_retriever(similarity_top_k=100) # Increased retrieval pool
        semantic_results = retriever.retrieve(expanded_query)
        
        # Combine with keyword scoring
        combined_results = []
        forced_includes = []
        
        # Add forced activities based on intents (V3 logic)
        if intents:
            for intent in intents:
                if intent['activities']:
                    forced_includes.extend(intent['activities'])
        
        for node in semantic_results:
            metadata = node.metadata
            activity_code = metadata.get('code', '')
            
            semantic_score = node.score * 100
            kw_score = self.keyword_score(
                query,
                metadata.get('keywords', ''),
                metadata.get('activity_name', '')
            )
            
            hybrid_score = (
                semantic_score * WEIGHTS['semantic'] +
                kw_score * WEIGHTS['keyword']
            )
            
            # Multi-intent boost (V3 logic)
            if activity_code in forced_includes:
                hybrid_score += 30
            
            # Category boost (V3 logic)
            if category_hint and metadata.get('category') == category_hint:
                hybrid_score += WEIGHTS['category_boost']
            
            # Phrase match bonus (V3 logic)
            query_lower = query.lower()
            keywords_lower = metadata.get('keywords', '').lower()
            
            if query_lower in keywords_lower:
                hybrid_score += 40
            else:
                query_words = query_lower.split()
                if len(query_words) >= 2:
                    for i in range(len(query_words) - 1):
                        two_word = ' '.join(query_words[i:i+2])
                        if two_word in keywords_lower:
                            hybrid_score += WEIGHTS['phrase_match_bonus']
                            break
            
            combined_results.append({
                'code': metadata.get('code', ''),
                'activity_name': metadata.get('activity_name', ''),
                'category': metadata.get('category', ''),
                'group': metadata.get('group', ''),
                'description': metadata.get('description', ''),
                'keywords': metadata.get('keywords', ''),
                'risk_rating': metadata.get('risk_rating', 'Low'),
                'related_activities': metadata.get('related_activities', ''),
                'when': metadata.get('when', ''),
                'third_party': metadata.get('third_party', ''),
                'semantic_score': round(semantic_score, 2),
                'keyword_score': round(kw_score, 2),
                'final_score': round(min(hybrid_score, 100), 2),
                'is_multi_intent': activity_code in forced_includes
            })
        
        # Sort by final_score (V3 logic)
        combined_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return combined_results

    # --- MODIFIED: The Main Search Function for V4 ---
    def search(self, query: str, top_k: int = 5) -> Dict:
        """Performs the full hybrid search with multi-intent and expert reranking."""
        
        # 0. CHECK CACHE FIRST
        print(f"🔍 Checking cache for: '{query}'")
        query_embedding = self._get_query_embedding(query)
        cache_hit, cached_result, similarity = self._check_cache(query, query_embedding)
        
        if cache_hit:
            # Return cached result with cache metadata
            cached_result['from_cache'] = True
            cached_result['cache_similarity'] = similarity
            return cached_result
        
        print(f"❌ No cache hit (best similarity: {similarity:.2%}). Running full search...")
        
        # 1. Multi-Profile Intent Detection (V4 LLM)
        start_llm = time.time()
        intent_data = self._get_intent_and_profiles(query)
        print(f"   LLM Intent Extraction Time: {time.time() - start_llm:.2f}s")
        
        # Robustly extract profile data
        primary_profile_data = intent_data.get('primary_profile', {})
        secondary_profile_data = intent_data.get('secondary_profile', {})
        
        # Extract intents and ensure they are a list
        primary_intents = primary_profile_data.get('intents', [])
        secondary_intents = secondary_profile_data.get('intents', [])
        
        all_intents = primary_intents + secondary_intents
        
        # Ensure category_hint is set for V3 logic (if it failed to be extracted)
        category_hint = primary_profile_data.get('category_hint') 
        # CRITICAL FALLBACK: If the LLM failed to return 'Financial', force it based on high-risk keywords
        if not category_hint and any(kw in query.lower() for kw in ['invoice', 'purchase', 'pay', 'debt', 'acquire']):
            category_hint = "Financial"
        elif not category_hint:
            category_hint = "Services" # Safe default fallback
        
        # 2. Run the Existing V3 Hybrid Scoring
        initial_scored_results = self._run_hybrid_search_and_score(query, all_intents)

        # 3. Expert Reranking and Final Scoring
        if USE_TWO_STAGE_RERANKING:
            # TWO-STAGE RERANKING (Fast)
            print(f"🚀 Using Two-Stage Reranking (Fast Mode)")
            
            # Stage 1: Quick filter with GPT-4o-mini
            start_quick = time.time()
            quick_filtered = self._quick_filter_with_mini(query, initial_scored_results[:QUICK_FILTER_POOL])
            print(f"   Stage 1 (Quick Filter): {time.time() - start_quick:.2f}s")
            
            # Stage 2: Deep analysis with GPT-4o on top candidates only
            start_deep = time.time()
            top_candidates = quick_filtered[:DEEP_ANALYSIS_POOL]
            reranked_results = self._rerank_with_llm(query, top_candidates)
            print(f"   Stage 2 (Deep Analysis): {time.time() - start_deep:.2f}s")
            
        else:
            # ORIGINAL SINGLE-STAGE RERANKING (Accurate but slow)
            print(f"⚠️ Using Single-Stage Reranking (Original Mode)")
            RERANK_POOL_SIZE = 75 
            rerank_candidates = initial_scored_results[:RERANK_POOL_SIZE]
            
            start_rerank = time.time()
            reranked_results = self._rerank_with_llm(query, rerank_candidates) 
            print(f"   LLM Reranking Time: {time.time() - start_rerank:.2f}s")
        
        # Calculate the Ultimate Score
        for result in reranked_results:
            llm_score = result.get('llm_match_score', 0)
            hybrid_score = result.get('final_score', 0)
            
            # Ensure scores are numeric before calculation
            try:
                llm_score = float(llm_score)
                hybrid_score = float(hybrid_score)
            except ValueError:
                llm_score = 0.0
                hybrid_score = 0.0
                
            # --- NEW PENALTY LOGIC ---
            adjusted_hybrid_score = hybrid_score
            
            # If the Expert LLM score is below the threshold, heavily penalize the V3 Hybrid Score.
            if llm_score < LLM_PENALTY_THRESHOLD:
                adjusted_hybrid_score *= LLM_PENALTY_MULTIPLIER
            # --- END PENALTY LOGIC ---
            
            result['ULTIMATE_SCORE'] = (llm_score * ULTIMATE_WEIGHTS['llm_match_score']) + \
                                        (adjusted_hybrid_score * ULTIMATE_WEIGHTS['hybrid_score'])
            
        # Final Sorting by ULTIMATE_SCORE
        reranked_results.sort(key=lambda x: x['ULTIMATE_SCORE'], reverse=True)

        # --- NEW: QUALITY FILTER - Only keep activities with LLM score >= 60% ---
        qualified_results = [r for r in reranked_results if r.get('llm_match_score', 0) >= 60]
        
        # Split into initial display (up to top_k) and additional (the rest)
        results_initial = qualified_results[:top_k]
        results_additional = qualified_results[top_k:]
        
        # Add rank to ALL qualified results (for consistent numbering)
        for idx, result in enumerate(qualified_results):
            result['rank'] = idx + 1
        
        # Confidence logic (based on the new ULTIMATE_SCORE of the top match)
        confidence = 'none'
        if results_initial:
            if results_initial[0]['ULTIMATE_SCORE'] >= 85:
                confidence = 'high'
            elif results_initial[0]['ULTIMATE_SCORE'] >= 65:
                confidence = 'medium'
        else:
            confidence = 'low'
        
        result = {
            'query': query,
            'confidence': confidence,
            'primary_profile': primary_profile_data,
            'secondary_profile': secondary_profile_data,
            'results_initial': results_initial,
            'results_additional': results_additional,
            'total_qualified': len(qualified_results),
            'total_shown_initially': len(results_initial),
            'from_cache': False,  # This is a fresh search
            'cache_similarity': 0.0
        }
        
        # Save to cache for future use
        self._add_to_cache(query, query_embedding, result)
        
        return result