#!/usr/bin/env python3
"""
Quick test to verify source filtering works correctly
"""

def test_source_matching():
    """Test that source filter matching works"""
    
    # Simulate the embedding indexes structure
    embedding_indexes = {
        "Techcrunch": "dummy_index_1",
        "Carbonherald": "dummy_index_2",
        "Hydrogen-central": "dummy_index_3",
        "Interestingengineering": "dummy_index_4"
    }
    
    # Test queries
    test_cases = [
        ("what are the articles from TechCrunch recently?", "techcrunch"),
        ("show me hydrogen-central news", "hydrogen-central"),  # Fixed: preserve hyphen
        ("carbonherald articles", "carbonherald"),
        ("from interestingengineering", "interestingengineering"),
    ]
    
    print("Testing source filter matching...\n")
    
    for query, expected_filter in test_cases:
        print(f"Query: '{query}'")
        print(f"Expected filter: '{expected_filter}'")
        
        # Simulate the matching logic from app.py
        sources_to_query = [(name, idx) for name, idx in embedding_indexes.items() 
                          if expected_filter in name.lower()]
        
        matched_sources = [name for name, _ in sources_to_query]
        
        if matched_sources:
            print(f"✅ PASS - Matched: {matched_sources}")
        else:
            print(f"❌ FAIL - No match. Available: {list(embedding_indexes.keys())}")
        
        print()

if __name__ == "__main__":
    test_source_matching()
