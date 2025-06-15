OPTIMIZED_EXTRACTION_PROMPT = """Extract ALL significant information from this text chunk. Be exhaustive and precise.

TEXT CHUNK:
{full_text}

EXTRACT EVERYTHING IMPORTANT AS JSON:

{
  "quotes": [
    {
      "id": "q1",
      "text": "exact quote or key statement",
      "author": "speaker/writer or document",
      "context": "brief context",
      "page": page_number_if_available,
      "importance": "high|medium|low",
      "entity_ids": ["e1"],
      "metric_ids": ["m1"]
    }
  ],
  "entities": [
    {
      "id": "e1", 
      "name": "entity name",
      "type": "person|organization|place|product|concept|technology",
      "description": "what this entity is/does",
      "importance": "high|medium|low",
      "quote_ids": ["q1"],
      "metric_ids": ["m1"]
    }
  ],
  "metrics": [
    {
      "id": "m1",
      "value": "123.45",
      "unit": "%|$|years|units|etc",
      "type": "percentage|currency|date|quantity|ratio|count",
      "context": "what this number means",
      "significance": "high|medium|low",
      "entity_ids": ["e1"],
      "quote_ids": ["q1"]
    }
  ],
  "relations": [
    {
      "source_entity_id": "e1",
      "target_entity_id": "e2",
      "type": "works_for|owns|partners_with|competes_with|located_in|influences",
      "description": "how they relate",
      "strength": "strong|moderate|weak"
    }
  ]
}

CRITICAL RULES:
1. Extract EVERY quote, statement, or significant text
2. Extract EVERY person, company, place, product, concept mentioned
3. Extract EVERY number, percentage, date, statistic, measurement
4. Create cross-references: if a metric appears in a quote, link them
5. Don't miss anything - be comprehensive
6. Use sequential IDs: q1,q2,q3... e1,e2,e3... m1,m2,m3...
7. Return ONLY valid JSON, no markdown, no explanations

Focus on COMPLETENESS - extract everything of value from this chunk."""