"""
Business Activity Matching Engine V3
Cloud-compatible version with Qdrant Cloud
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

# Get credentials from environment/secrets
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Configuration
SYNONYM_FILE = "synonym_dictionary.json"
CSV_FILE = "enhanced_activities_manual.csv"
COLLECTION_NAME = "business_activities"

# Scoring weights
WEIGHTS = {
    'semantic': 0.35,
    'keyword': 0.65,
    'phrase_match_bonus': 25,
    'category_boost': 20
}

# Setup
Settings.embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
Settings.llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")


class ActivityMatcherV3:
    def __init__(self):
        """Initialize the matching engine with Qdrant Cloud"""
        print("🔧 Loading matching engine V3 (Cloud)...")
        
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
        with open(SYNONYM_FILE, 'r') as f:
            self.synonyms = json.load(f)
        
        # Load activities CSV
        print("   Loading activities data...")
        self.df = pd.read_csv(CSV_FILE, encoding='latin-1')
        
        print("✅ Matching engine V3 ready!\n")
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
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
        """Detect likely category from query"""
        query_lower = query.lower()
        
        category_keywords = {
            'ICT': ['software', 'website', 'app', 'technology', 'IT', 'cloud', 'digital', 'cyber', 'coding', 'programming', 'data', 'internet', 'online platform', 'web', 'tech', 'computer', 'system', 'network', 'development'],
            'F&B': ['restaurant', 'cafe', 'food', 'catering', 'kitchen', 'dining', 'bakery', 'eatery', 'coffee shop', 'bar', 'pub', 'bistro', 'canteen', 'cafeteria', 'food service'],
            'Rentals': ['rental', 'rent', 'lease', 'leasing', 'hiring', 'letting', 'accommodation', 'property rental', 'equipment rental', 'vehicle rental', 'space rental'],
            'Trading': ['import', 'export', 'wholesale', 'trading', 'goods', 'retail', 'shop', 'store', 'sell', 'sell online', 'digital sales', 'e-commerce', 'online selling', 'commodity', 'dealer', 'supplier', 'distributor'],
            'Professional': ['consulting', 'advisory', 'legal', 'accounting', 'marketing', 'consultancy', 'lawyer', 'management', 'audit', 'tax', 'strategy', 'business services', 'professional services', 'bookkeeping', 'financial advice', 'corporate', 'attorney'],
            'Manufacturing': ['production', 'manufacturing', 'factory', 'fabrication', 'making', 'assembly', 'processing', 'industrial production', 'textile manufacturing', 'garment production', 'metal works'],
            'Healthcare': ['medical', 'clinic', 'healthcare', 'dental', 'hospital', 'pharmacy', 'doctor', 'health', 'medicine', 'patient care', 'nursing', 'physiotherapy', 'laboratory', 'diagnostic', 'surgery', 'treatment'],
            'Transportation': ['transport', 'logistics', 'shipping', 'cargo', 'delivery', 'courier', 'freight', 'moving', 'haulage', 'trucking', 'vehicle', 'taxi', 'bus', 'marine transport', 'air cargo', 'warehousing', 'tour packages', 'tour operator'],
            'Education': ['training', 'education', 'school', 'courses', 'institute', 'learning', 'teaching', 'coaching', 'fitness coaching', 'tutoring', 'academy', 'university', 'college', 'kindergarten', 'nursery', 'workshop'],
            'Administrative': ['administration', 'administrative support', 'office management', 'secretarial', 'business support', 'document preparation', 'data entry', 'filing', 'clerical', 'tour operator', 'travel agency', 'event management'],
            'Financial': ['finance', 'banking', 'investment', 'insurance', 'loan', 'credit', 'financial services', 'money exchange', 'forex', 'stock', 'securities', 'fund management', 'wealth management', 'mortgage'],
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
    
    def detect_multiple_intents(self, query: str) -> List[Dict]:
        """Detect multiple business intents in query"""
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
    
    def keyword_score(self, query: str, keywords: str, activity_name: str) -> float:
        """Calculate keyword matching score"""
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
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Hybrid search with multi-intent detection
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Detect multiple intents
        intents = self.detect_multiple_intents(query)
        
        if len(intents) > 1:
            print(f"   🎯 Detected {len(intents)} business intents:")
            for intent in intents:
                print(f"      - {intent['name']}: '{intent['keyword']}'")
        
        # Expand query
        expanded_query = self.expand_query(query)
        
        # Detect category
        category_hint = self.detect_category(query)
        if category_hint:
            print(f"   Detected category: {category_hint}")
        
        # Semantic search
        retriever = self.index.as_retriever(similarity_top_k=top_k * 6)
        semantic_results = retriever.retrieve(expanded_query)
        
        # Combine with keyword scoring
        combined_results = []
        forced_includes = []
        
        # Add forced activities based on intents
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
            
            # Multi-intent boost
            if activity_code in forced_includes:
                hybrid_score += 30
                print(f"   ⭐ Boosted: {metadata.get('activity_name', '')} (multi-intent)")
            
            # Category boost
            if category_hint and metadata.get('category') == category_hint:
                hybrid_score += WEIGHTS['category_boost']
            
            # Phrase match bonus
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
        
        # Sort
        combined_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Ensure forced activities are in top
        top_results = []
        forced_results = [r for r in combined_results if r['is_multi_intent']]
        other_results = [r for r in combined_results if not r['is_multi_intent']]
        
        top_results.extend(forced_results[:3])
        remaining = top_k - len(top_results)
        top_results.extend(other_results[:remaining])
        
        # Add rank
        for idx, result in enumerate(top_results):
            result['rank'] = idx + 1
        
        # Confidence
        if not top_results:
            confidence = 'none'
        elif top_results[0]['final_score'] >= 80:
            confidence = 'high'
        elif top_results[0]['final_score'] >= 60:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'query': query,
            'confidence': confidence,
            'category_hint': category_hint,
            'intents': intents,
            'results': top_results,
            'total_found': len(top_results)
        }


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Business Activity Matching Engine V3                   ║
    ║   Multi-Intent Detection + Hybrid Matching               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    matcher = ActivityMatcherV3()
    
    # Test query
    test_query = "I want to sell mobile phones online"
    print(f"\n🔍 Test query: '{test_query}'\n")
    result = matcher.search(test_query, top_k=3)
    
    for r in result['results']:
        print(f"   ✓ {r['activity_name']} ({r['final_score']:.1f}%)")
    
    print("\n✅ Engine is working!")


if __name__ == "__main__":
    main()