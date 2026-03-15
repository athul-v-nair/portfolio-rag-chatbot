RESUME_PARSER_PROMPT = """
You are a resume parser.

Extract structured information from the resume.

Return ONLY valid JSON.

Format:
<format>
{{
 "summary": "",
 "contact": "",
 "skills": [],
 "projects": [
   {{
     "title": "",
     "description": "",
     "technologies": []
   }}
 ],
 "experience": [
   {{
     "company": "",
     "role": "",
     "description": ""
   }}
 ],
 "education": [
   {{
     "college_name": "",
     "course_name": "",
     "related courses": ""
   }}
 ],
}}
</format>

<rules>
Rules:
- Do not invent information other than specified in resume block
- Keep descriptions concise
- Technologies should be extracted if mentioned
</rules>

Resume:
<resume>
{resume_text}
</resume>
"""