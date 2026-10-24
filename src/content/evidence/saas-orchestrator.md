---
title: "The SaaS Orchestrator"
subtitle: "Multi-tenant backend for governed AI pipelines"
summary: "The productization layer for the Leadership Agent: tenant isolation, authentication, role boundaries, and audit logging built into the platform rather than the application."
status: "real-system"
claims:
  - "Tenant isolation enforced by PostgreSQL Row Level Security, not application logic"
  - "ES256/JWKS token verification with platform_admin, tenant_admin, editor, and viewer roles"
  - "Audit logging for every run, stage, and QC result"
links:
  - { label: "Explore the architecture", href: "/#innovation-lab", kind: "section" }
questions:
  - "Why enforce tenant isolation in the database instead of the application?"
  - "What would a compliance reviewer check first in this design?"
order: 4
draft: false
---
