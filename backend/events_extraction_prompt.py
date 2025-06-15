EVENTS_EXTRACTION_PROMPT = """EXTRACT ALL EVENTS FROM THIS TEXT. Focus on temporal information and everything that happened.

{full_text}

Extract these 3 categories - prioritize EVENTS:

EVENTS - Extract ALL temporal occurrences:
• Actions that happened at specific times
• Milestones, achievements, announcements
• Decisions made, meetings held, agreements signed  
• Product launches, releases, publications
• Changes in status, appointments, departures
• Financial events (funding, acquisitions, earnings)
• Market events, crises, opportunities
• Any occurrence with temporal significance

ENTITIES - Extract ALL people, organizations, products mentioned:
• People (names, titles, roles)
• Organizations (companies, institutions, agencies)
• Places (countries, cities, locations)
• Products (software, tools, systems, brands)
• Concepts (methodologies, frameworks, technologies)

METRICS - Extract ALL numbers and measurements:
• Dates and timeframes (most important for events)
• Numbers with meaning (percentages, amounts, counts)
• Statistics and measurements
• Financial figures, performance indicators

JSON FORMAT:
{
  "events": [
    {
      "id": "ev1",
      "title": "brief event title",
      "description": "detailed description of what happened",
      "temporal_marker": "2023-Q4" | "January 2023" | "last year" | "recently" | "before the merger",
      "date_parsed": "2023-01-15" | null,
      "certainty": "certain" | "estimate",
      "type": "milestone" | "announcement" | "decision" | "launch" | "meeting" | "change" | "financial" | "market",
      "entity_ids": ["e1", "e2"],
      "metric_ids": ["m1"],
      "supporting_text": "exact text from PDF that supports this event",
      "page_number": 5,
      "importance": "high" | "medium" | "low"
    }
  ],
  "entities": [
    {
      "id": "e1", 
      "name": "entity name",
      "type": "person" | "organization" | "place" | "product" | "concept",
      "description": "what/who they are",
      "event_ids": ["ev1", "ev2"],
      "metric_ids": ["m1"]
    }
  ],
  "metrics": [
    {
      "id": "m1",
      "value": "123",
      "unit": "%" | "$" | "years" | "Q4" | etc,
      "type": "date" | "percentage" | "currency" | "quantity",
      "context": "what it measures",
      "event_ids": ["ev1"],
      "entity_ids": ["e1"]
    }
  ]
}

CRITICAL RULES FOR EVENTS:
✓ Every event MUST have supporting_text (exact quote from PDF)
✓ Every event MUST have page_number where it was found
✓ temporal_marker: use the EXACT temporal reference from the text
✓ date_parsed: only if you can determine a specific date (YYYY-MM-DD format)
✓ certainty: "certain" if explicitly stated, "estimate" if implied/approximate
✓ Look for: "in 2023", "last quarter", "recently", "after the acquisition", "before launch"
✓ Extract ALL events - even small ones might be important
✓ Cross-reference: link events to entities and metrics

GOAL: Create a complete timeline of everything that happened, with full traceability back to the source text."""