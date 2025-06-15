EXTRACTION_PROMPT = """Analiza el siguiente documento y extrae información estructurada según este modelo de datos:

{full_text}

OBJETIVO: Transformar este PDF en conocimiento estructurado para ahorro de trabajo cognitivo.

EXTRAE LA SIGUIENTE INFORMACIÓN:

1. QUOTES (citas textuales importantes, declaraciones, afirmaciones clave):
   Formato: {
     "id": "q1", "q2", etc.
     "text": "la cita textual exacta",
     "author": "quien lo dijo o escribió",
     "context": "contexto donde aparece la cita",
     "page": número de página (si está disponible),
     "entity_ids": ["e1", "e2"], // entidades mencionadas EN la cita
     "metric_ids": ["m1", "m2"]  // métricas mencionadas EN la cita
   }

2. ENTITIES (personas, organizaciones, lugares, productos, conceptos clave):
   Formato: {
     "id": "e1", "e2", etc.
     "name": "nombre completo de la entidad",
     "type": "person" | "organization" | "place" | "product" | "concept",
     "description": "descripción breve pero informativa",
     "importance": "high" | "medium" | "low",
     "quote_ids": ["q1", "q3"], // quotes donde se menciona
     "metric_ids": ["m1", "m4"], // métricas asociadas a esta entidad
     "attributes": {} // atributos adicionales específicos
   }

3. METRICS (números, estadísticas, fechas, porcentajes, cantidades, KPIs):
   Formato: {
     "id": "m1", "m2", etc.
     "value": "el valor numérico o fecha",
     "unit": "unidad de medida (%, $, años, etc)",
     "type": "percentage" | "currency" | "date" | "quantity" | "ratio",
     "context": "qué representa esta métrica y por qué es importante",
     "entity_ids": ["e1"], // entidades asociadas
     "quote_ids": ["q2"], // quotes donde aparece
     "trend": "increasing" | "decreasing" | "stable" | null
   }

4. RELATIONS (relaciones explícitas entre entidades):
   Formato: {
     "source_entity_id": "e1",
     "target_entity_id": "e2", 
     "type": "owns" | "works_for" | "competes_with" | "partners_with" | "located_in" | etc,
     "description": "descripción de la relación",
     "strength": "strong" | "moderate" | "weak"
   }

REGLAS IMPORTANTES:
- CADA quote importante debe estar capturada
- CADA entity mencionada debe estar registrada
- CADA metric (número con significado) debe estar extraída
- Las referencias cruzadas (entity_ids, metric_ids, quote_ids) son CRÍTICAS
- Si una métrica aparece en una quote, debe estar en ambos lados
- Si una entity se menciona en una quote, debe estar referenciada

OBJETIVO FINAL: El usuario debe poder entender TODO el documento sin leerlo, solo navegando por estas estructuras interconectadas.

Devuelve un JSON válido con las 4 categorías: quotes, entities, metrics, relations."""