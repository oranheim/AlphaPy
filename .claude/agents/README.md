# AlphaPy Agent Organization

## Agent Ecosystem Overview

This directory contains specialized agents with proper Claude Code frontmatter. Each agent can be activated using `/agents <name>` in Claude Code.

## ✅ **Functional Agents** (With Frontmatter)

### 🏗️ **software-architect** 
**Command**: `/agents software-architect`  
**Use for**: System design, architecture decisions, dependency planning
- Code architecture analysis and design decisions
- Dependency management and build system optimization  
- Performance optimization and scalability planning

### 🔬 **ml-engineer**
**Command**: `/agents ml-engineer`  
**Use for**: ML algorithms, feature engineering, model optimization
- ML algorithm implementation and tuning
- Feature engineering and selection
- Trading strategy development

### 💰 **fintech-specialist**
**Command**: `/agents fintech-specialist`  
**Use for**: Trading systems, market data, portfolio management
- Trading system design and implementation
- Market data acquisition and validation
- Portfolio optimization and risk management

### ✅ **qa-engineer**
**Command**: `/agents qa-engineer`  
**Use for**: Testing, code quality, CI/CD validation
- Test strategy design and implementation
- Code quality validation (linting, typing, formatting)
- CI/CD pipeline optimization

### 🚀 **devops-engineer**
**Command**: `/agents devops-engineer`  
**Use for**: CI/CD, deployment, infrastructure, monitoring
- CI/CD pipeline design and maintenance
- Development environment standardization
- Dependency management and security

### 👁️ **code-reviewer**
**Command**: `/agents code-reviewer`  
**Use for**: General code reviews and best practices
- Code quality assessment and improvement suggestions
- General security vulnerability identification
- Performance optimization recommendations

### 💎 **system-developer** ⭐ **(Foundation Expert)**
**Command**: `/agents system-developer`  
**Use for**: Enterprise-grade system engineering and software architecture
- Robust system design, performance optimization, reliability engineering
- Production-grade library architecture, resource management, concurrency
- Algorithmic trading infrastructure requirements, real-time constraints
- **Foundation expert** - use for core system engineering and architecture

### 🔒 **security-reviewer** ⭐ **(Security Expert)**
**Command**: `/agents security-reviewer`  
**Use for**: Expert ML/FinTech security and compliance review
- Deep security analysis (OWASP Top 10, PCI DSS, GDPR)
- ML-specific security (model serialization, data poisoning)
- Financial compliance and regulatory requirements
- **Security expert** - use for production security validation

## 🤖 **Agent Autonomy & Collaboration**

Each agent now has **autonomous collaboration rules** embedded within their configuration. Agents will:
- **PROACTIVELY delegate** to other specialists when needed
- **Autonomously decide** when to hand off tasks
- **Collaborate directly** with other agents using explicit agent names
- **Maintain their specialty** while knowing when they need help

## 🚀 **How Agent Dancing Works**

### **Autonomous Workflows**
Agents automatically collaborate using these patterns:
1. **Analysis** → **Implementation** → **Review** → **Validation**
2. **Specialists delegate** to each other autonomously
3. **Security review** happens automatically for production code
4. **Testing validation** is built into all workflows

### **Example Agent Collaboration**
```
User: "Fix the remaining test failures"
└── qa-engineer (analyzes issues)
    └── delegates to ml-engineer (implements fixes)
        └── requests security-reviewer (security validation)
            └── coordinates with qa-engineer (final validation)
```

### **Proactive Delegation Examples**
- **`ml-engineer`** → "Trading logic needed - delegating to fintech-specialist"
- **`qa-engineer`** → "Implementation fixes required - delegating to ml-engineer"
- **`software-architect`** → "Security review needed - requesting security-reviewer"

## 📋 **Quick Agent Selection**

### **Start With These Agents**
- **System Engineering** → `/agents system-developer` ⭐ (Foundation)
- **Architecture/Design** → `/agents software-architect`
- **ML Implementation** → `/agents ml-engineer`
- **Trading Systems** → `/agents fintech-specialist`
- **Testing Issues** → `/agents qa-engineer`
- **CI/CD Problems** → `/agents devops-engineer`
- **Security Review** → `/agents security-reviewer` ⭐ (Security)

### **Agents Will Autonomously**
✅ **Recognize their limits** and delegate when needed  
✅ **Coordinate directly** with other specialists  
✅ **Maintain context** through handoffs  
✅ **Ensure quality** through automatic reviews  
✅ **Handle security** through built-in compliance checks  

## 🎯 **Result: Self-Organizing Team**

No manual orchestration required! Just activate the appropriate starting agent and watch them:
- **Collaborate automatically** with other specialists
- **Maintain high quality** through built-in reviews
- **Ensure security** through automatic compliance checks
- **Deliver complete solutions** through coordinated teamwork

**The agents now dance together!** 🕺💃