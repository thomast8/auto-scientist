"""Example domain knowledge template.

Write domain-specific context that helps the agents understand the science.
This gets injected into all agent prompts.
"""

EXAMPLE_DOMAIN_KNOWLEDGE = """\
## Domain: [Your Domain Name]

### Background
[Describe the scientific domain, key concepts, and what you're trying to model]

### Data
[Describe the dataset: what it contains, how it was collected, key variables]

### Key Physics / Theory
[Describe the theoretical framework, key equations, physical constraints]

### Known Challenges
[What makes this problem hard? Known pitfalls, identifiability issues, etc.]

### Key Metrics
[What metrics should the model optimize? How do you define success?]
"""
