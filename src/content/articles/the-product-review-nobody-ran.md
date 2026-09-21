---
title: "The Product Review Nobody Ran"
description: "Recruiters closed twice as many hires while hiring still got slower. A feature-by-feature audit of what's actually broken, and what the evidence supports."
pubDate: 2026-09-21
ogImage: "/assets/articles/the-product-review-nobody-ran-card.jpg"
ogImageAlt: "Abstract illustration of a broken pipeline and dashboard, representing an article auditing the hiring process like a failing product"
tags: ["enterprise-transformation", "delivery"]
draft: true
---

*Recruiters got faster while the hiring system they run got slower. That gap says more about how the product is built than about the people working inside it.*

## The Review Nobody Ran

Every part of your hiring system can look like it's winning. Recruiters closing more roles than ever. Screening moving faster. Sourcing sharper than it's been in years. And somehow, filling a single seat still takes longer than it did two years ago.

That contradiction is the whole story, and it's worth sitting with before the numbers make it official.

Start with the dashboard.

Between 2023 and 2025, in the largest hiring dataset available, applications per job rose 29 percent, from 189 to 244. Over those same two years, the average number of recruiters per organization fell 36 percent, from 7.23 to 4.62.

The recruiters who remained got dramatically more productive. Monthly hires per recruiter more than doubled, from 2.4 to 4.9.

And the system-level metric still got worse. Time-to-fill climbed from just over 47 days to nearly 60.<sup><a href="#source-1" id="ref-1">[1]</a></sup>

Put that dashboard in a product review.

Demand: up. Operating capacity: down. Individual throughput: up. End-to-end cycle time: worse.

Any organization that actually treats hiring, and the people it brings in, as a valuable and necessary resource wouldn't look at those numbers and quietly congratulate each function for hitting its local KPI. It would ask why the outcome that mattered got worse while every component appeared to improve. That's the review hiring rarely gets.

There's an important caveat before going any further. The dataset comes from a recruiting-software company that sells products intended to improve hiring, and its public benchmark doesn't clearly establish that every organization in the sample is U.S.-based. A separate benchmark reported by SHRM found time-to-fill moving in the opposite direction, from 48 days in 2023 to 41 in 2024, though the two benchmarks draw on different underlying populations and aren't a clean apples-to-apples comparison.<sup><a href="#source-2" id="ref-2">[2]</a></sup>

So the point isn't that the national hiring process now takes exactly 60 days. Nobody needs a precise number to see the problem. The point is this: inside one of the largest hiring datasets that exists, the people running the pipeline got dramatically better at their jobs, and the pipeline got worse anyway.

If this were software, blaming the engineers wouldn't fix anything. Every one of them would say the same true thing: their part of the system is working exactly as designed. So you'd go looking for whoever owns the architecture instead.

There isn't one.

## The Org Chart Nobody Drew

Hiring is not a single-user app, nor is it a simple two-sided market. Stop treating it like a transactional department. It is a multi-sided enterprise platform with competing users, stretched operators, unmeasured owners, hard regulatory guardrails, and cascading downstream costs. When candidates, recruiters, managers, and compliance teams each optimize for their own piece of the process, the problem is not that any single group is failing. It is that no one is managing the system, and once you look at the architecture, the failure becomes obvious.

### The Candidate

The candidate is one primary user of the system. The analogy has limits. Candidates are supplying labor, being evaluated, negotiating an employment relationship, and protected by employment law. They are not simply consumers clicking through an app.

But their behavior still produces familiar product signals: conversion, abandonment, trust, friction, and willingness to continue. Seventy-one percent of job seekers in one 2025 survey expected an application to take less than 30 minutes. Thirty-five percent said they would abandon an application that took too long.<sup><a href="#source-3" id="ref-3">[3]</a></sup> In a separate 2026 survey, 38 percent said they had already withdrawn from a hiring process because it involved an AI interview.<sup><a href="#source-4" id="ref-4">[4]</a></sup> That is user behavior. But candidates are not the only users whose needs matter.

### The Hiring Manager

The hiring manager is the product owner of the role, except nobody bothered to tell them, give them a spec sheet, or onboard them to the build. Their job is supposed to begin long before a résumé touches an inbox: define what "good" looks like, set clear criteria, make the hard trade-offs, and keep the goalposts from moving mid-stream.

When organizations commit to skills-based hiring, NACE's research shows real, if uneven, follow-through: 87 percent apply it in interviews, 81 percent in job descriptions, 65 percent in screening, and 58 percent in formal rubrics.<sup><a href="#source-5" id="ref-5">[5]</a></sup> That's evidence the role matters, not evidence that any one organization applies it consistently across all four. What's missing entirely is telemetry on how the role actually behaves. The available research doesn't say how often requirements shift mid-search, how many weeks a file sits waiting on a decision, or how much of that delay is genuine deliberation versus something else entirely. The role is critical to the architecture. Its actual operation is a black box of unmeasured delay.

### The Recruiter

The recruiter is the pipeline operator, and they are the clearest case of local-metric theater: the one productivity measure tied to their part of the system improved sharply, and the product still got worse. Between 2023 and 2025, as recruiting teams shrank by more than a third, the recruiters who remained more than doubled their monthly hires, from 2.4 to 4.9.<sup><a href="#source-1" id="ref-1b">[1]</a></sup> They didn't just meet their metrics; they crushed them. They optimized their way right off the org chart.

Yet the overall cycle time still climbed toward two months. Asking recruiters to "move faster" is the executive equivalent of yelling at the air traffic controller while the runway is blocked by a parked plane. The operators did their jobs. They cleared their stage. Whatever's actually broken sits somewhere else in the system.

### The Team Waiting on the Other Side

Then there is the ghost user in hiring analytics: the actual team left holding the bag. An open headcount isn't a vacant box on a PowerPoint slide; it is operational debt. Someone covers the extra customer calls. Someone absorbs the midnight escalations. Someone shelves a critical initiative because the seat remains empty.

The striking part isn't that this happens. The striking part is how little of it gets measured. Time-to-fill gets tracked closely. The burnout, delivery risk, and cost absorbed by the people doing two jobs while the seat stays open don't show up in any of the sources reviewed here.

No single role owns the outcome. Candidates own their application status. Managers own the hiring decision. Recruiters operate the pipeline. Compliance owns the risk controls. The receiving team inherits the fallout. Every node in the network behaves rationally, hits its local SLA, and pushes the cost onto a team with zero vote in the process.

## The Feature Audit

A product review eventually leaves the org chart and starts inspecting the experience. Where does the product actually break?

### Intake: The Front Door Optimizes for Volume

Intake is flooded, and the pipeline treats volume like progress. Applications per job surged from 116 in 2022 to 244 in 2025, more than doubling against that baseline.<sup><a href="#source-1" id="ref-1c">[1]</a></sup>

What's driving the flood, candidate AI tools, one-click application buttons, or something else entirely, is genuinely unclear. There's no verified figure for how much of the growth traces to either. What does exist is a broader number: 31 percent of U.S. job seekers in one 2025 survey said they used AI somewhere in their job search, which isn't the same claim as AI-driven application volume.<sup><a href="#source-3" id="ref-3b">[3]</a></sup> What is clear is the real failure: we built a front door instrumented to count volume, with no mechanism built to measure signal.

The system tracks every file received. It cannot tell you if a single one is worth reading. Volume exploded. Signal died.

### Flow: Local Efficiency, Systemic Gridlock

The middle of the funnel contains the ultimate paradox. Between 2023 and 2025, recruiter output doubled while average time-to-fill dragged out to nearly 60 days.<sup><a href="#source-1" id="ref-1d">[1]</a></sup>

It would be convenient to blame excessive interview rounds or indecisive managers. The available evidence does not support that conclusion. Time-to-fill can absorb scheduling delays, approval chains, shifting requirements, compensation negotiations, freezes, candidate withdrawals, interview structure, and any number of other dependencies.

Anyone who has led an enterprise transformation recognizes this pattern immediately. Security hits its SLA. Procurement hits its SLA. Legal hits its SLA. And the new hire sits in the dark for weeks waiting for access.

Nothing has to be individually broken for the product to fail completely. When every node in a workflow optimizes for its own local throughput, end-to-end delivery grinds to a halt. It isn't a broken feature. It's an unmanaged network.

### Communication: The System Ghosting Its Own Users

For a meaningful share of candidates, the product simply stops reporting state. Fifty-one percent of candidates who completed an AI-driven interview in one 2026 survey said they never heard back.<sup><a href="#source-4" id="ref-4b">[4]</a></sup>

Candidates spend hours uploading employment history, answering screening prompts, and recording video responses. Then the system drops into total radio silence.

Every delivery app on earth sends real-time status updates. The failure here isn't technical. The product knows its own state. It's simply choosing not to share it.

### Evaluation: Unenforced Rubrics

Structure exists on paper, but falls apart in the build. Among employers already using skills-based hiring, 58 percent say they apply it specifically to interview rubrics.<sup><a href="#source-5" id="ref-5b">[5]</a></sup>

That's adoption, not enforcement. A rubric nobody checks risks becoming organizational decoration instead of an actual evaluation system. Dashboards can confirm a rubric was attached to a requisition. They offer no observability into whether interviewers scored independently, whether opinions got anchored by the loudest voice in the debrief, or whether the framework got quietly abandoned the moment a charismatic candidate walked in.

The test suite was written. Nobody has confirmed anyone's running the build.

### Governance: Control Is a Feature, Not a Bug

Not every piece of friction is waste, and deleting every gate in the name of speed is a fatal mistake.

An EEOC case alleging that automated screening software had rejected more than 200 qualified applicants over 55 ended in a $365,000 settlement.<sup><a href="#source-7" id="ref-7">[7]</a></sup> It demonstrated the real risk of unmonitored automation: a bad decision rule doesn't disappear at scale, it scales right alongside everything else. A 2024 academic study examining 391 employers subject to New York's AEDT bias-audit and notice law found only 18 had posted audit reports and 13 had posted transparency notices, though the authors caution that non-posting doesn't prove noncompliance, since employers retain real discretion over whether a given tool falls in scope. That's a single preprint, not a compliance census, but it's a real signal that the governance layer around these tools remains mostly unbuilt.<sup><a href="#source-6" id="ref-6">[6]</a></sup>

Compliance and algorithmic governance aren't red tape standing in the way of delivery. They are non-negotiable architectural guardrails. The goal isn't tearing down the gates to shave a few days off the timeline. It's integrating governance into the workflow so the product stays legally defensible at speed.

## The Diagnosis

The five failures look different. They are not.

The front door measures volume better than signal. The funnel measures stage activity better than total flow. The communication layer knows process state without reliably exposing it. Evaluation frameworks exist without enough evidence that they govern actual decisions. Governance controls exist outside workflows that were often designed without them.

The common problem is not a lack of data. It is fragmented ownership and fragmented measurement. The parts of the system are measured. The whole of it isn't.

That distinction matters because metrics shape behavior. If recruiters are measured on throughput, throughput improves. If managers are rewarded for avoiding a bad hire, adding another interview can look locally rational. If compliance owns risk independently, another approval can look locally rational. If candidates can apply to ten jobs in the time it once took to apply to one, maximizing optionality can look locally rational too.

Every actor can make a defensible decision from inside their part of the system. Collectively, those decisions can create a product nobody would intentionally design. That is the failure. Hiring is not simply broken because people are bad at hiring. It behaves like a collection of locally optimized services that nobody owns as an end-to-end product.

## The Metric the Review Should Use

Saying "time-to-fill is the wrong metric" is not enough. Replacing it with an equally vague phrase like "quality of hire" does not solve the problem either. The system needs a product-level outcome.

Call it time to validated hire: not a single magic number, but a north-star outcome supported by three measures.

- **Speed.** How long did it take from approved need to accepted hire?
- **Candidate experience.** Did the people moving through the process receive a reasonable, transparent experience with clear communication and proportionate effort?
- **Downstream outcome.** Did the person hired succeed after joining, using whatever role-appropriate performance, retention, or ramp measure the organization already trusts?

None of the research reviewed for this article provides a standardized national formula combining those three dimensions. That is the point. Organizations are extremely good at measuring activity inside hiring and surprisingly inconsistent at measuring whether the hiring product worked.

A north-star outcome would not eliminate the supporting metrics. Recruiter productivity would still matter. Time-to-fill would still matter. Application conversion would still matter. Compliance would still matter. But they would become diagnostic measures underneath a product outcome instead of becoming the outcome themselves.

## What Ships First

A real review ends with changes, owners, and tests. Not slogans.

### Product change: Stage the front door

Do not demand everything from every candidate before establishing basic fit. Seventy-one percent of job seekers in the Employ survey expected an application to take less than 30 minutes, while 35 percent said they would abandon one that took too long.<sup><a href="#source-3" id="ref-3c">[3]</a></sup> Ask for the information required to make the next decision. Collect the rest when it becomes necessary.

There is no strong published before-and-after study establishing an ideal staged-application design, so treat this as a product hypothesis. Instrument it. Test abandonment, qualified-candidate conversion, and downstream quality.

### Product change: Expose process state

Every applicant should know the application was received. Candidates who complete meaningful interview stages should know when the process ends.

The case for this does not depend on proving that automated status communication increases quality-of-hire. The existing failure is sufficiently clear: 51 percent of respondents who completed an AI interview in the 2026 Greenhouse survey said they never heard back.<sup><a href="#source-4" id="ref-4c">[4]</a></sup> The system already knows its state. Expose it.

### Process change: Structure evaluation before increasing automation

The strongest causal evidence in this evidence base comes from a 2025 academic paper reporting two field experiments at a recruitment platform. The one that matters most here is the large-scale operational deployment: more than 34,000 unique candidates applied for a single Junior Frontend Engineer position, randomly assigned either to a traditional résumé-screening process or to a structured process that opened with an AI-assisted standardized interview before the same blinded human evaluation.<sup><a href="#source-8" id="ref-8">[8]</a></sup>

In that deployment, using the unadjusted result with no-shows counted as failures, candidates from the structured track passed their final interview at 54 percent, compared with 34 percent from the traditional track. A separate part of the same paper found that candidates who completed the AI interview, whether or not they passed it, were more likely to report a new job five months later than candidates who never completed it.<sup><a href="#source-8" id="ref-8b">[8]</a></sup>

Those findings deserve caution. The research is a preprint. The specific hiring employer isn't named, only the platform they used. The population was one junior technical role. U.S.-only applicability is not established. The two tracks also surfaced somewhat different candidate populations. So the result should not be stretched into "AI interviewing works" as a general claim. The more defensible lesson is narrower: changing the structure of evaluation before intake can change measurable outcomes. That is a much stronger starting point than automating an evaluation process nobody has first made consistent.

### Process change: Lock the decision rules before the first interview

Define the criteria. Define who evaluates which criteria. Define how evidence will be scored. Define what triggers another interview. Then measure whether the process was actually followed.

NACE's findings show structured rubrics are already being used at meaningful scale.<sup><a href="#source-5" id="ref-5c">[5]</a></sup> What remains unproven is how faithfully those rubrics govern real decisions, or whether a specific cap on interview rounds improves outcomes. So do not publish an arbitrary number of "ideal interviews." Run the experiment internally.

### Operating-model change: Give someone end-to-end ownership

Recruiting can own recruiting operations. Hiring managers can own selection. Compliance can own regulatory controls. None of those roles automatically owns the product. Someone needs authority to ask the uncomfortable cross-functional question: why did every team hit its metric while the overall outcome got worse?

There is no published evidence establishing that assigning a single hiring-product owner improves organizational outcomes. This is an operating hypothesis. But without end-to-end ownership, every other change risks becoming another local optimization.

### Governance change: Build the gate into the road

Do not remove controls designed to reduce discriminatory or unaccountable decision-making. Integrate them. Automated decisions should be auditable. Required disclosures should be part of workflow design. Human escalation paths should exist where consequential automated decisions require review.

The EEOC case demonstrates the reason plainly: automation can scale a flawed rule just as efficiently as it scales a good one.<sup><a href="#source-7" id="ref-7b">[7]</a></sup> Governance should not disappear. It should stop arriving as a surprise.

## The Review Outcome

Now return to the dashboard. Applications increased. Recruiter capacity decreased. Recruiter throughput more than doubled. And in the same benchmark, end-to-end time-to-fill got worse.<sup><a href="#source-1" id="ref-1e">[1]</a></sup>

None of those numbers is meaningless. They are simply incomplete. The mistake is treating a collection of component metrics as proof that the product works, when not one of them is time to validated hire, the outcome nobody on the org chart above actually owns.

Hiring doesn't need another argument about whether recruiters are too slow, candidates apply too broadly, managers are too selective, or AI is saving or destroying the process. Those debates focus on individual actors. The more useful question is architectural: what outcome is the hiring product actually designed to produce, who owns that outcome, and what evidence would prove the next change made it better?

That is what a real product review would ask. Define the outcome. Instrument the system around it. Ship the smallest credible change. Then measure the result. If the outcome didn't move, the ticket isn't done.

---

## Sources

1. <span id="source-1">Greenhouse, ["The Hire Standard: Hiring Benchmarks 2026"](https://www.greenhouse.com/recruiting-benchmarks), March 2026. 6,000+ companies and 640M+ applications, 2022–2025. Vendor-owned platform telemetry; public preview does not clearly establish a U.S.-only sample.</span> [↑](#ref-1)

2. <span id="source-2">SHRM, ["Recruiters Express Optimism for 2025"](https://www.shrm.org/topics-tools/news/talent-acquisition/recruiters-express-optimism-for-2025), January 2, 2025. Secondary reporting on Recruiter Nation vendor research; underlying methodology was not fully visible in the surfaced material.</span> [↑](#ref-2)

3. <span id="source-3">Employ, ["Employ's Latest Report on Job Seeker Insights Helps Recruiters Fine-Tune Processes..."](https://www.employinc.com/news_item/employs-latest-report-on-job-seeker-insights-helps-recruiters-fine-tune-processes-personalize-outreach-and-stay-ahead-of-hiring-trends/), April 29, 2025. Survey of 1,500+ U.S. employed or actively job-seeking adults. Vendor-funded survey.</span> [↑](#ref-3)

4. <span id="source-4">Greenhouse, ["63% of Job Seekers Have Faced an AI Interview. Most Haven't Had a Good One Yet."](https://www.greenhouse.com/newsroom/63-of-job-seekers-have-faced-an-ai-interview-most-havent-had-a-good-one-yet), May 1, 2026. 2,950 active job seekers internationally, including 1,200 U.S. respondents; findings stated as U.S.-specific unless otherwise noted. Vendor-funded survey.</span> [↑](#ref-4)

5. <span id="source-5">National Association of Colleges and Employers, ["Employer Use of Skills-Based Hiring Practices Grows"](https://www.naceweb.org/job-market/trends-and-predictions/employer-use-of-skills-based-hiring-practices-grows), January 12, 2026. Job Outlook 2026 survey on hiring new college graduates, fielded August 7 to September 22, 2025; 183 respondents (170 NACE employer members, a 22.7% response rate among eligible members, plus 13 nonmember companies). Scoped to new-college-graduate hiring, not employers generally.</span> [↑](#ref-5)

6. <span id="source-6">Lucas Wright et al., ["Null Compliance: NYC Local Law 144 and the Challenges of Algorithm Accountability"](https://arxiv.org/abs/2406.01399), June 3, 2024. Academic preprint examining 391 employers; primary empirical research, not a government compliance census.</span> [↑](#ref-6)

7. <span id="source-7">U.S. Equal Employment Opportunity Commission, ["iTutorGroup to Pay $365,000 to Settle EEOC Discriminatory Hiring Suit"](https://www.eeoc.gov/newsroom/itutorgroup-pay-365000-settle-eeoc-discriminatory-hiring-suit), September 11, 2023. Primary federal enforcement source documenting the agency's allegations and settlement.</span> [↑](#ref-7)

8. <span id="source-8">Ada Aka, Emil Palikot, Ali Ansari, and Nima Yazdani, ["Better Together: Quantifying the Benefits of AI-Assisted Recruitment"](https://arxiv.org/abs/2507.08029), July 8, 2025, revised August 7, 2026 (v2). Reports two field experiments at a recruitment platform; the larger, a live pipeline deployment, covers more than 34,000 unique applicants for a single junior technical role. Academic preprint; the specific hiring employer is unnamed and U.S.-only applicability is not established.</span> [↑](#ref-8)

*Note on source quality: the strongest recurring descriptive numbers in this piece, including recruiter throughput and time-to-fill, come from a single vendor's platform telemetry, and that vendor sells tools designed to improve the hiring process. A competing benchmark reports time-to-fill moving in the opposite direction over part of the same period. The randomized recruitment study provides stronger causal evidence but has narrower applicability and remains a preprint. Where the available evidence establishes a gap rather than a finding, this article treats that gap as such rather than filling it with a plausible-sounding number.*

---

**About the author**

Reginal Campbell writes about enterprise technology, AI governance, leadership, and the systems organizations build to make consequential decisions.

[Read more articles](/articles) · [Connect with Reginal on LinkedIn](https://www.linkedin.com/in/reginal-campbell-pmp-1551845/)
