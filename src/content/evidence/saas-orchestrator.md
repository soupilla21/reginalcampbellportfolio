---
title: "The SaaS Orchestrator"
subtitle: "Multi-tenant backend for governed AI pipelines"
summary: "The productization layer for the Leadership Agent: tenant isolation, authentication, role boundaries, and audit logging built into the platform rather than the application."
status: "verified"
kind: "built-system"
corpusKeys: ["saas_orchestrator_multitenant_backend", "saas_orchestrator_phase_5_commercialization"]
claims:
  - "Tenant isolation enforced by PostgreSQL Row Level Security, not application logic"
  - "ES256/JWKS token verification with platform_admin, tenant_admin, editor, and viewer roles"
  - "Audit logging for every run, stage, and QC result"
  - "Described as audit-ready; not represented as holding any formal certification"
  - "Billing, usage metering, and self-service onboarding are planned, not built"
links:
  - { label: "Explore the architecture", href: "/portfolio#innovation-lab", kind: "section" }
questions:
  - "Why enforce tenant isolation in the database instead of the application?"
  - "What would a compliance reviewer check first in this design?"
order: 5
draft: false
---
