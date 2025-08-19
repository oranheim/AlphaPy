# Agent Definition Consolidation Plan

Based on AGENT_ANALYSIS.md principles and individual agent review.

## Issues Found Across All Agents

### 1. **Project State References** (MUST REMOVE)
- **code-reviewer.md**: "125 ruff errors"
- **devops-engineer.md**: "125/130 ruff errors, 95/95 tests"
- **qa-engineer.md**: "100% test success rate, 95/95 tests, 130 ruff errors"
- **fintech-specialist.md**: Specific performance metrics
- Others likely have similar issues

### 2. **Unnecessary Fields** (REMOVE)
- All agents have `color:` field - not used, should remove

### 3. **Missing Elements** (ADD)
- "When Invoked" section - immediate actions
- Success criteria (generic, not project-specific)
- Clear handoff protocol structure

### 4. **Good Patterns to Keep**
- Collaboration rules (all agents have these)
- Domain expertise sections
- Primary responsibilities
- Tool/command references (where applicable)

## Standardized Structure (Per AGENT_ANALYSIS.md)

```markdown
---
name: agent-name
description: One-line description. When to use this agent.
model: sonnet
---

# Role
You are a [Senior/Expert] [Role] with [X+ years] experience in [specific domain].

# Primary Responsibilities
1. Core responsibility
2. Core responsibility
3. Core responsibility
4. Core responsibility

# Domain Expertise
- Area of expertise
- Technical capabilities
- Specialized knowledge

# When Invoked
Immediately:
1. Assess the current situation
2. Identify the specific task
3. Begin work without asking for clarification

# Workflow
1. **Analyze**: [Specific steps]
2. **Implement/Review/Design**: [Core action]
3. **Verify**: [Validation steps]
4. **Handoff**: [If needed]

# Success Criteria
- [ ] Task completed as requested
- [ ] Quality standards maintained
- [ ] Changes verified
- [ ] Handoff prepared if needed

# Collaboration
## When to Delegate
- Condition → Delegate to `agent-name`
- Condition → Delegate to `agent-name`

## My Core Role
[One sentence describing what I do vs delegate]
```

## Consolidation Steps

### Phase 1: Remove Project State (HIGH PRIORITY)
1. Remove all specific numbers (test counts, error counts)
2. Replace with generic quality statements
3. Remove hardcoded paths and versions

### Phase 2: Standardize Structure (MEDIUM)
1. Add "When Invoked" section to all
2. Standardize headers and hierarchy
3. Ensure consistent markdown formatting

### Phase 3: Cleanup (LOW)
1. Remove `color:` field from all agents
2. Balance detail levels
3. Ensure collaboration rules are clear

## Specific Changes Per Agent

### code-reviewer.md
- Remove: "125 ruff errors" reference
- Remove: color field
- Add: "When Invoked" section
- Keep: Review checklist (excellent pattern)

### devops-engineer.md
- Remove: "125/130 ruff errors, 95/95 tests"
- Remove: color field
- Simplify: "Current Infrastructure" section
- Keep: Command reference section

### fintech-specialist.md
- Remove: Specific performance metrics
- Remove: color field
- Add: "When Invoked" section
- Keep: Domain knowledge structure

### ml-engineer.md
- Remove: color field
- Add: "When Invoked" section
- Keep: ML conventions section

### qa-engineer.md
- Remove: "100% test success rate, 95/95 tests, 130 errors"
- Remove: color field
- Replace: Specific numbers with generic quality standards
- Keep: Testing strategy section

### security-reviewer.md
- Check for project state references
- Remove: color field if present
- Standardize structure

### system-developer.md
- Add: model field (missing)
- Remove: color field if present
- Consider: Reducing detail level for consistency

### software-architect.md
- Remove: Any project state references
- Remove: color field
- Standardize structure

## Implementation Order

1. **First**: Remove all project state references (prevents confusion)
2. **Second**: Remove color fields (cleanup)
3. **Third**: Add missing sections (When Invoked, Success Criteria)
4. **Fourth**: Standardize formatting and structure

## Validation

After consolidation, each agent should:
- ✅ Have no project-specific metrics
- ✅ Follow standardized structure
- ✅ Be self-contained and autonomous
- ✅ Have clear collaboration rules
- ✅ Focus on role, not current state