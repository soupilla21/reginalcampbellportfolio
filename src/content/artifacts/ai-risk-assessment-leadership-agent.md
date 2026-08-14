---
title: "AI Risk Assessment: The Leadership Agent"
description: "A completed risk assessment of a real 9-agent content pipeline, including the scoring rationale, three revised scores, and the residual risks I chose to accept."
ogImage: "/assets/artifacts/ai-risk-assessment-leadership-agent-card.jpg"
ogImageAlt: "Governance Artifact title card: AI Risk Assessment: The Leadership Agent, Reginal Campbell, reginalcampbell.com"
lastReviewed: 2026-08-13
version: "1.0"
artifactType: "risk-assessment"
frameworkAlignment: ["nist-ai-rmf", "iso-42001", "eu-ai-act"]
anchorScenario: "leadership-agent"
order: 1
draft: false
---

This is a risk assessment I actually ran on a system I built and still operate.

It is not a template with the answers removed. The scores are the scores I assigned, including three I changed after taking a harder look at the evidence. The residual risk section also names the risks I chose to accept instead of pretending I solved them.

I am publishing this because the assessment format is not the hard part. Anyone can download a scoring matrix.

The hard part is the judgment inside it.

What tier does a system belong in when the frameworks do not give you a clean answer? How do you defend a score you cannot prove with certainty? When do you decide a control is good enough? And are you willing to document the risk you chose to accept and put your name next to it?

That reasoning is usually invisible. It sits in a spreadsheet or governance deck nobody outside the organization ever sees.

This is what it looks like when the reasoning stays visible.

One caveat about self-assessment deserves to be stated up front. I am the builder, operator, and assessor of this system. That creates a structural weakness in the assessment itself, not just in the system. It appears again in the residual risk section because that is where it belongs. I have not tried to explain it away.

## 1. System characterization

**What it is.** The Leadership Agent is a nine-agent autonomous pipeline that ingests public source material, scores it for relevance, drafts long-form professional content, runs it through an automated quality gate, and surfaces finished drafts for human approval. It replaced a $10K/month ghostwriting dependency and reduced manual editorial intervention by roughly 40% during its first 90 days of operation.

**Deployment.** Local-first. Inference runs on local hardware. There is no third-party model API call at any point in generation. Persistence is Postgres, with Row Level Security enforced at the database layer rather than through application logic. A misconfigured route therefore cannot leak data across tenant boundaries.

**Data flows.** There are four, and being precise about them matters because the risk profile of this system is largely determined by what data crosses what boundary.

| Flow | Source | Destination | Crosses an external boundary? |
| --- | --- | --- | --- |
| Ingest | Public RSS and web sources | Local scoring store | Inbound only |
| Generation | Local store + brand guide | Local inference | No |
| Feedback | Approved output patterns | Brand guide | No |
| Publication | Approved draft | External platform | Outbound, after human approval |

The critical property is that no proprietary or personal input data leaves the local environment because it never enters a system capable of sending it anywhere.

That is not a policy statement someone could violate through a configuration mistake. It is an architectural property.

That distinction is why the confidentiality risks in the register score as low as they do. The scores are defensible because the exposure path has been removed, not because a policy says nobody should use it.

**Autonomy level.** Partial autonomy with a mandatory terminal human gate. The pipeline runs end to end without intervention through ingest, scoring, drafting, editing, and QC. Then it stops. Nothing publishes without a human action.

In NIST AI RMF terms, the system has high operational autonomy but no independent decision authority over its consequential output.

**Where a human is actually in the loop.** There are three points, and only three.

That precision matters because "human in the loop" has become one of the most overused phrases in AI governance. Too often it means someone technically could intervene, even though nobody consistently does.

1. **Terminal approval before publication.** Genuine, blocking, and exercised on every output. The human reads the draft and either publishes it or does not. The entire risk position depends on this control.
2. **Source list curation.** Periodic, not per run. The human decides which sources the pipeline is allowed to ingest. It is a real control, but one exercised infrequently enough that drift is possible.
3. **Brand guide review.** Nominally a human control. In practice, the Feedback Ingester writes to the brand guide automatically after a pattern appears three times, and the human sees the result only if they review it. This is the weakest of the three controls and, as section 4 shows, where the assessment changed most.

There is no human intervention between ingest and draft.

A source can be scored, selected, synthesized, and incorporated into a draft without anyone seeing it until the finished text appears.

That is intentional. It is where much of the time saving comes from.

It also means every error introduced across that span is caught at one point or not at all.

## 2. Inherent risk tier determination

The inherent risk determination, before controls, is **Tier 2: Moderate**, using an internal three-tier scale:

- Tier 1: Elevated
- Tier 2: Moderate
- Tier 3: Routine

**The case for Tier 3: Routine, which is the tier I rejected.**

The argument for Tier 3 is legitimate:

- The system makes no decision about a person. It does not screen, score, rank, allocate, or deny anything to anybody.
- It processes no personal or regulated data. Its inputs are public.
- It is single-operator. There is no customer or outside party relying directly on its output.
- Its output is technically reversible. A published post can be deleted.
- It has no external model dependency, removing an entire category of third-party data exposure risk.
- Under the EU AI Act, it falls into none of the Annex III high-risk categories and is not a prohibited practice.

Using the tests most frameworks emphasize, such as whether a system touches personal data, makes consequential decisions about people, or operates in a regulated use case, Tier 3 is defensible.

I still rejected it.

**Why? Three reasons.**

First, most standard tests are designed around systems that act *on* people.

This system acts *as* a person.

Its output is published under a real identity, in the first person, into a professional market. The primary harm model is not, "The system made the wrong decision about someone."

It is, "The system said something false in my voice, and I published it."

Frameworks built primarily around consequential decision-making do not have a natural category for that risk. That is a limitation of the framework, not evidence that the risk is insignificant.

Second, the audience changes the impact.

The readers include hiring managers, peers, and practitioners evaluating professional judgment and competence. A fabricated statistic in a generic marketing blog is embarrassing. A fabricated statistic in material being used as evidence of professional judgment can damage credibility much more seriously.

The content may also remain indexed, cached, or screenshotted long after the original is deleted.

So while the output is technically reversible, reputational exposure is much less reversible in practice.

Third, and most importantly, the system contains an unsupervised loop that modifies its own instructions.

The Feedback Ingester promotes patterns into the brand guide. The brand guide shapes future generation.

That means the system does not simply execute the same rules repeatedly. Its future behavior is partly a function of its own history.

That is categorically different from a stable system.

Tier 3 is appropriate for systems whose behavior remains predictable within a fixed control structure. This one can change the instructions shaping its own output.

That alone moves it above routine.

**Tier 1 was not seriously considered.**

There is no safety, health, fundamental-rights, or financial exposure, and no third party depends on the output. Calling a system elevated when it is not weakens the meaning of the category for systems that actually warrant it.

**On the EU AI Act specifically.**

This system is not high-risk under Annex III, and I do not want to imply otherwise. Inflating a classification is its own form of governance failure.

The obligation that applies in substance is transparency around AI-generated content through the Article 50 family of disclosure requirements.

The control is straightforward and appears in the register as R7: published content is disclosed as AI-assisted.

I have treated that as an applicable obligation rather than spending energy debating whether a single-operator system falls within every technical boundary of the provision. In this case, the argument costs more than the compliance.

## 3. Scored risk register

Likelihood and impact are scored from 1 to 5.

**Risk score = Likelihood × Impact**

Bands:

- 1 to 6: Low
- 7 to 12: Moderate
- 13 to 19: Elevated
- 20 to 25: Critical

The Revised column reflects reassessment after reviewing the evidence. Section 4 explains each change.

Three scores moved.

One moved down.

| ID | Risk | Initial (L×I) | Revised (L×I) | Band |
| --- | --- | --- | --- | --- |
| R1 | Confident factual fabrication published under my name | 3×4 = 12 | **3×5 = 15** | Elevated |
| R2 | Feedback loop entrenches a defect into the brand guide | 2×3 = 6 | **4×4 = 16** | Elevated |
| R3 | Source material reproduced too closely without attribution | 2×4 = 8 | No change | Moderate |
| R4 | Homogenization: output converges on a narrow voice | 3×2 = 6 | No change | Low |
| R5 | Silent pipeline failure produces incomplete output | 4×2 = 8 | **2×2 = 4** | Low |
| R6 | Confidentiality: proprietary input leaves the environment | 1×5 = 5 | No change | Low |
| R7 | Undisclosed AI authorship | 2×3 = 6 | No change | Low |
| R8 | Assessor independence: builder, operator, and assessor are one person | 5×3 = 15 | No change | Elevated |
| R9 | Local model ceiling produces lower-quality reasoning than a frontier model | 3×3 = 9 | No change | Moderate |

A note on R6 scoring 1×5.

If proprietary data left the environment, the impact could be severe, so impact stays at 5.

Likelihood is 1 rather than 0 because zero-risk claims usually create more confidence than they deserve.

But the reason likelihood is 1 instead of 3 is architectural, not procedural. There is no external inference call to misconfigure.

This is one of the clearest examples in the register of why architecture is stronger than policy. A control that cannot be bypassed through an ordinary mistake should score differently from one that can.

R8 at 5×3 also deserves explanation.

Likelihood is 5 because this is not something that might happen. It is a permanent property of the current operating model.

The risk remains elevated and is not mitigated. It is accepted.

That is why it appears again in section 5.

## 4. What changed, and why

The three score revisions are probably the most important part of this assessment.

An assessment where every initial score survives contact with the evidence should raise questions.

### R2: Feedback loop entrenchment, 6 to 16

This was the largest revision and the one that changed the assessment most.

My initial score treated the Feedback Ingester primarily as a quality mechanism with a manageable drift risk.

Then I stopped relying on my memory of how it worked and read the promotion logic carefully.

A pattern promotes after three appearances.

There is no human confirmation step.

There is no decay on promoted patterns.

There is no review automatically triggered by promotion.

To the system, a pattern appearing three times because it is genuinely useful looks exactly like a pattern appearing three times because an upstream defect caused repetition.

I had already seen the second case happen.

UTM-suffixed URLs in the RSS ingest created duplicate scored items, which produced repeated themes downstream. The problem was eventually diagnosed and fixed at the ingest boundary.

But the more important governance lesson came later.

The feedback loop does not simply inherit upstream defects. It can amplify them.

A duplicate item does more than waste a slot. It creates the appearance of recurrence, which is exactly the signal the promotion logic uses to decide something should influence future behavior.

Likelihood moved from 2 to 4 because there was direct evidence of the mechanism firing.

Impact moved from 3 to 4 because a promoted defect is not a single bad output. It becomes a persistent instruction influencing everything generated afterward, and the degradation can happen quietly.

### R1: Factual fabrication, 12 to 15

Likelihood stayed at 3.

Impact moved from 4 to 5.

My original framing treated the primary consequence as reputational embarrassment.

That was too narrow.

This content is being read by people evaluating professional judgment. In that context, a confidently stated false claim is not merely an editorial error. It can become evidence someone uses to judge competence and credibility.

It also survives deletion through indexing, caching, and screenshots.

Likelihood did not change because the terminal human gate is genuine and is exercised.

But likelihood and impact are different variables.

I had allowed confidence in the control to suppress the impact score.

That is a scoring error.

A strong control can reduce likelihood. It does not make the consequence less serious if the control fails.

### R5: Silent pipeline failure, 8 to 4

This score moved down.

The Writer and Editor agents required timeouts above 180 seconds for complex content. At 90 seconds, runs could disappear silently. There was no error, no alert, and no completed output, while the system still appeared healthy.

That incident drove the original likelihood score of 4.

After increasing the timeout and adding explicit run-status logging, 90 days of logs showed no recurrence.

More importantly, the failure mode changed.

A failed run now fails visibly instead of quietly producing the appearance of normal operation.

Likelihood therefore dropped to 2 based on evidence.

I am including a downward revision deliberately.

If every risk assessment only moves scores upward, it stops measuring risk and starts performing caution.

Scores should move in whichever direction the evidence supports.

Being willing to lower one is part of what makes raising another one credible.

## 5. Controls, and what they actually cost

Every control below is currently in place.

I included the cost column because generic control catalogues usually do not.

That omission matters.

Controls create friction. When that friction is ignored during design, people bypass the control later because the operational cost was never acknowledged.

A control that looks perfect on paper but cannot survive real workflow pressure is not a strong control.

| Control | Risks addressed | What it actually costs |
| --- | --- | --- |
| **Terminal human approval before publication.** Blocking. No output publishes without a human reading it end to end. | R1, R3, R4, R7 | This is the expensive control. It consumes a meaningful share of the roughly 40% editorial saving the system was built to create. The control and the value proposition are directly in tension. It also does not scale because throughput remains capped by one person's attention. I accept that because R1 has an impact score of 5 and no automated structural check can reliably identify a confident falsehood. |
| **Publisher QC gate.** Automated. Checks word-count thresholds, banned phrases, structural compliance, and CTA presence. Failures are logged with specific reasons. | R4, R5 | It catches form, not truth. A beautifully structured fabrication can pass. The bigger operational risk is false rejection. Too many bad rejections teach the operator to override the gate, and habitual overrides can destroy a control while leaving the appearance that the control still exists. Override frequency is therefore something I watch. |
| **Local-only inference.** Architectural. No external model API exists in the generation path. | R6 | Hardware cost and a lower model ceiling than a frontier API. That trade is not neutral because it increases R9. The system buys confidentiality partly by spending accuracy. |
| **RLS enforced at the database layer.** Postgres-level, not application logic. | R6 | Slower schema iteration because policies must be written and tested alongside structural changes. The cost is worth it because an incorrectly configured route cannot create a cross-tenant leak. |
| **Data contracts enforced at the ingest boundary.** Tracking parameters are stripped during ingest, not during deduplication. | R2, R5 | Every new source type requires boundary rules before it is enabled, which slows source expansion. This is also the control that would have prevented the duplicate-item defect behind the R2 revision. |
| **Explicit run-status logging and timeout floors.** | R5 | Additional log volume and periodic review. The weakness is obvious: a logging control provides little value if nobody reads the logs. That means the control can degrade through the same human inattention it was designed to address. |
| **AI-assistance disclosure on published output.** | R7 | Effectively zero cost. Cheap controls still deserve documentation. |
| **Quarterly brand-guide diff review.** Added as a direct result of this assessment. The promotion threshold was raised, and every promoted pattern is reviewed against the previous quarter's guide. | R2 | The loop adapts more slowly. That is a real cost because automatic promotion was producing useful improvements. The control also depends on a recurring human commitment, which makes it inherently weaker than an architectural control. It reduces R2. It does not eliminate it. |

Notice what is not in this control set.

There is no mechanism independently verifying factual claims.

No fact-checking agent.

No retrieval-grounded citation requirement.

No external verification layer.

That is a deliberate gap.

It appears below as residual risk instead of being disguised as a control that does not exist.

## 6. Residual risk accepted

These risks are named, owned, and tied to explicit review triggers.

They are not open remediation items.

They are decisions to continue operating despite known exposure.

I own all three, which is itself part of the problem with the first one.

### RR1: Assessor independence, from R8

The builder, operator, and assessor of this system are the same person.

There is no independent review of my scoring, and I am structurally motivated to view my own system favorably.

No internal control can fully remove that problem. A single-operator environment cannot manufacture genuine segregation of duties.

Publishing the assessment openly is a partial substitute. External scrutiny creates a weak form of independence, but I do not confuse it with actual independent review.

That is one reason this document is public instead of sitting in a spreadsheet on my machine.

**Owner:** Reginal Campbell

**Review trigger:** Any second operator, any third party relying on the output, or any use of this system on behalf of an employer or client.

Any one of those changes the risk from uncomfortable to unacceptable and requires an independent reassessment.

### RR2: No factual verification layer, from R1

The system can produce a fluent, confident, false claim.

The only thing standing between that claim and publication is one human reading carefully.

I accept that exposure today because adding a retrieval-grounded verification stage would introduce latency and complexity that I consider disproportionate for a single-operator content system, and because the terminal gate is genuinely exercised rather than nominal.

But the risk needs to be described accurately.

This control depends on the operator's attention remaining strong.

Human attention degrades.

**Owner:** Reginal Campbell

**Review trigger:** The first published factual error that reaches an external reader, or any increase in publication cadence that materially reduces review time per output.

Either event moves this risk from accepted to unacceptable.

### RR3: Residual feedback-loop drift below the detection threshold, from R2

The quarterly diff review can identify pattern-level drift.

It is much weaker at identifying a series of small, individually reasonable changes that gradually compound into a voice I did not intentionally choose.

Slow drift is exactly the kind of failure a quarterly snapshot can miss.

**Owner:** Reginal Campbell

**Review trigger:** Two consecutive quarterly reviews showing promoted patterns I do not recognize as deliberate, or any single promoted pattern traced to a data defect rather than a genuine editorial signal.

## 7. The decision this assessment drove

The Leadership Agent runs local and on-premises because the data cannot leave the building.

That architectural decision predates this assessment.

I want to be precise about the sequence rather than construct a cleaner story than what actually happened.

The architecture came first as a judgment call.

The assessment tested that judgment afterward.

What changed was not the decision.

What changed were the terms of the decision.

The original instinct was straightforward: local-first buys confidentiality.

The assessment confirmed that.

R6 scores 1×5 because the exposure path is architecturally absent rather than procedurally prohibited. That is a stronger control position than any policy could create.

But the register also exposed something the original instinct did not fully account for.

Local-first *spends* accuracy to buy that confidentiality.

R9 exists because the local model ceiling is lower than a frontier API's.

The trade is not free.

And the cost lands directly against the risk with the highest impact score in the register.

That reframing produced the actual finding:

> Local-first inference and removal of the terminal human gate are mutually incompatible under the current risk model. The architecture eliminates a major confidentiality exposure but increases accuracy risk. The human gate is the only control in the current set directly addressing that accuracy exposure. As long as the architecture remains, the gate is not a temporary phase-one safeguard waiting to be automated away. It is load-bearing.

That matters because the most obvious next optimization for a system like this is to remove the bottleneck.

The bottleneck is the human.

The efficiency gain would be real, which makes the temptation real.

Without the assessment, removing the human gate could easily look like a straightforward maturity improvement.

It is not.

Doing so would remove the primary control against a risk the architecture itself helps increase.

The second major finding came from R2.

That finding changed the system, not just the document.

The promotion threshold was raised and the quarterly brand-guide diff review was added.

Neither existed before this assessment.

They exist because reading the promotion logic carefully instead of relying on memory moved a risk score from 6 to 16.

The full architecture, agent topology, and delivery outcomes are documented in the [Leadership Agent case study](/#ai-systems).

## 8. What this assessment deliberately does not cover

This section exists because silence around scope can look like oversight.

An assessment that appears to cover everything is usually less trustworthy than one that clearly states where it stops.

- **Model-level evaluation.** No benchmarking, bias testing, or capability evaluation of the underlying model. This assessment evaluates the system and its controls, not the model weights.
- **Security assessment.** No threat model, penetration test, or supply-chain review of the local stack. Those are important, adjacent concerns that require a different method.
- **Prompt injection through ingested content.** This is a legitimate risk for any system that ingests untrusted web content and passes it to a model. I have left it out because it belongs in the security assessment above. I would rather name the gap than give it a superficial score in the wrong document.
- **Business continuity.** Hardware failure, recovery, and availability are operational risks, not specifically AI risks.
- **Third-party or multi-tenant operation.** The system has a multi-tenant backend, but this assessment covers single-operator use only. Multi-tenant operation triggers RR1 and requires reassessment beginning with the tier determination.

## Sources

1. [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework), the Govern, Map, Measure, Manage structure used in this assessment.
2. [ISO/IEC 42001:2023, AI management systems](https://www.iso.org/standard/81230.html), the basis for the control and residual-risk framing.
3. [Regulation (EU) 2024/1689, the EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), including the Annex III risk categories and Article 50 transparency provisions referenced in section 2.

*Assessment method: NIST AI RMF structure, ISO/IEC 42001 control framing, and 5×5 likelihood-impact scoring. Constructed entirely from public frameworks and first-party system knowledge.*
