EXTRACTION_PROMPT_V3 = """EXTRACT ALL KEY INFORMATION FROM THIS TEXT. Miss nothing important.

{full_text}

Return JSON with these 4 categories. Be EXHAUSTIVE:

QUOTES - Extract ALL:
• Important statements, claims, conclusions
• Key insights, findings, recommendations  
• Direct quotes from people
• Significant definitions or explanations
• Critical facts or assertions

ENTITIES - Extract ALL:
• People (names, titles, roles)
• Organizations (companies, institutions, agencies)
• Places (countries, cities, locations)
• Products (software, tools, systems, brands)
• Concepts (methodologies, frameworks, technologies)

METRICS - Extract ALL:
• Numbers with meaning (percentages, amounts, counts)
• Dates and timeframes
• Statistics and measurements
• Financial figures
• Performance indicators
• Growth rates, ratios, comparisons

RELATIONS - Extract relationships between entities

JSON FORMAT:
{
  "quotes": [{"id":"q1", "text":"exact text", "author":"who said it", "context":"brief context", "page":1, "importance":"high", "entity_ids":["e1"], "metric_ids":["m1"]}],
  "entities": [{"id":"e1", "name":"entity name", "type":"person", "description":"what/who they are", "importance":"high", "quote_ids":["q1"], "metric_ids":["m1"]}],
  "metrics": [{"id":"m1", "value":"123", "unit":"%", "type":"percentage", "context":"what it measures", "significance":"high", "entity_ids":["e1"], "quote_ids":["q1"]}],
  "relations": [{"source_entity_id":"e1", "target_entity_id":"e2", "type":"works_for", "description":"relationship", "strength":"strong"}]
}

EXTRACTION RULES:
✓ Extract EVERYTHING of value - don't skip anything
✓ Include obvious AND subtle information  
✓ Cross-reference: link quotes↔entities↔metrics
✓ Use IDs: q1,q2... e1,e2... m1,m2...
✓ Return only valid JSON - no text before/after
✓ Be comprehensive - this chunk might contain critical info

GOAL: Extract ALL significant information so nothing important is lost."""