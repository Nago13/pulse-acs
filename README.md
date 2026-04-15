# Pulse — We built it for the places everyone else forgot.

> *"We didn't build Pulse for the billion people who already have AI. We built it for the places everyone else forgot."*

---

## The problem no one is talking about

Over 1 billion people now use AI every month.

But 2.2 billion people have no internet access at all — more than double.

While the world celebrates the AI revolution, it has quietly forgotten the people who need intelligent tools the most: communities in remote areas where doctors are scarce, connectivity is nonexistent, and health decisions are made with nothing but human memory and intuition.

---

## Brazil, up close

I'm from Brazil — the 8th largest economy in the world. A country that exports soybeans, iron ore, and now, increasingly, software.

And yet, in 2025, **only 37% of rural properties in Brazil have full 4G coverage**. Entire regions of the Amazon, the Northeast, the inland countryside — places where people live and get sick and need care — have no signal at all.

Brazil has a remarkable public health system called SUS — Sistema Único de Saúde — built on the principle that healthcare is a right, not a privilege. One of its most important pillars is the **Agente Comunitário de Saúde**, or ACS.

---

## Who is an ACS?

An ACS — Community Health Worker — is not a doctor. They don't diagnose. They don't prescribe.

They are a formal employee of Brazil's SUS — the public health system — trained, registered, and paid by the federal government to be the first point of contact between their community and the healthcare network. They are your neighbor: someone who lives in the same community, walks the same roads, knows the same families. Brazil has **over 200,000 of them**, each responsible for up to 750 people in their area.

Every week, they visit homes. They knock on doors. They sit with elderly patients, check on pregnant mothers, monitor children's growth. They carry a tablet or phone issued by their municipality, fill out digital forms using e-SUS — the government's official health records app — and report their findings in person at the local health center (UBS) at the end of each day or week.

That last part matters: **the ACS already goes back to the health center regularly.** That is exactly when Pulse syncs. No new infrastructure needed. No change in routine. The connection happens naturally, at the UBS, where internet is available — and everything collected offline during the day is sent to the supervising physician in one tap.

The gap is not the device. The gap is not the routine. The gap is what happens **between** visits — in the field, in homes scattered across rural Brazil where mobile signal is unreliable or simply nonexistent.

These are not edge cases. According to Brazil's IBGE, only 65.8% of rural households have any mobile network coverage — and that number has been **falling** since 2022, not growing. In absolute terms, roughly **3 million rural homes** have zero mobile coverage. These are exactly the homes the ACS visits.

What makes this even more striking: internet access in rural Brazil has actually grown significantly — from 35% of households in 2016 to 84.8% in 2024. People have Wi-Fi at home. They have internet at the health center. But the moment the ACS steps out the door and walks toward a family living down a dirt road, the signal disappears. The connectivity gap is not about homes — it is about the field.

That is where the ACS makes decisions alone — with no clinical support, no knowledge base, and no way to know what questions to ask next.

Until now.

---

## What Pulse does

Pulse is a **clinical copilot for community health workers** — not a diagnostic system.

It doesn't replace the ACS. It equips them.

When an ACS visits a patient, they describe what they observe in plain language. Pulse uses semantic search to find the most clinically similar conditions from a knowledge base of 75 diseases, then guides the worker through a progressive interview — suggesting the next best questions based on what's already been answered.

After each response, the system rebuilds the context and searches again, progressively narrowing the hypotheses. The ACS confirms or adjusts every suggestion. The decision is always human.

When the visit ends, everything is saved locally. When the ACS reaches the health center with internet, one tap syncs all records to the supervising physician — structured, legible, complete.

**The ACS arrives better prepared. The doctor receives better information. The patient gets better care.**

---

## Why offline-first is not a feature. It's the whole point.

Every AI health tool we found assumes connectivity. They work in hospitals. They work in cities. They fail the moment you step outside a signal range.

Pulse was designed from the ground up for the opposite reality.

The vector database runs locally via Docker. The semantic search model is cached on the device after first load. The clinical knowledge base lives entirely on the phone. No API calls. No cloud latency. No data sent anywhere without the health worker's explicit action.

This is what makes Pulse different from every other AI health tool:

**It works where it actually needs to work.**

---

## The technology

Pulse is built on **Actian VectorAI DB** — a vector database designed specifically for edge and offline deployment.

When an ACS describes a patient's symptoms, Pulse encodes that description into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`. The VectorAI DB compares this vector against 75 indexed clinical conditions and returns the most semantically similar results — understanding that *"fever with nightly chills after forest exposure"* and *"intermittent febrile episodes"* point to the same thing, without matching keywords.

The search is **hybrid**:
- **Semantic layer**: finds conditions whose clinical descriptions are meaningfully similar to what the ACS observed
- **Filter layer**: narrows results by geographic region and age group, because malaria is endemic in Amazônia but not in São Paulo

This is not keyword search. This is not a lookup table. This is a system that understands meaning — and does it entirely offline.

### Technical requirements met
**Hybrid Fusion** — semantic vector search combined with structured metadata filtering (Brazilian state, age group), merged into a single clinically-ranked result.

---

## Knowledge base

75 clinical conditions indexed, covering:

- Tropical and endemic diseases (dengue, malaria, leptospirosis, Chagas disease, leishmaniasis, schistosomiasis)
- Respiratory conditions (pneumonia, tuberculosis, asthma crisis, bronchiolitis)
- Maternal health (high-risk pregnancy, preeclampsia, puerperal infection, postpartum depression)
- Pediatric emergencies (severe malnutrition, febrile seizure, meningitis, severe dehydration)
- Chronic disease emergencies (hypertensive crisis, decompensated diabetes, heart failure)
- Mental health (suicide risk, acute psychosis, severe depression — with safe referral protocols)
- Rural injuries and poisonings (snakebite, agrochemical poisoning, burns, drowning)
- Neglected tropical diseases endemic to Brazil

Each condition includes:
- Clinical description in plain language (as an ACS would observe it)
- 4 key questions to guide the interview
- Suggested course of action by urgency level
- Geographic prevalence filters
- **Source traceability**: each entry is tagged with its origin (`MS-CAB`, `MS-GVS`, or `IA-pendente-validacao`) and a `validado_por` field for future medical validation

---

## The visit flow
```
ACS opens Pulse on their phone (no internet needed)
↓
Enters patient data: name, age, sex, state, chronic conditions
↓
Describes what they observe in plain language
↓
Pulse searches the vector database → returns top 3 hypotheses + first questions
↓
ACS asks the questions → enters responses
↓
Pulse rebuilds context vector → searches again → refines hypotheses
↓
Loop continues until ACS ends the interview
↓
Pulse suggests a course of action → ACS confirms or adjusts
↓
Visit saved locally as structured JSON
↓
At the health center with internet: one tap syncs to supervising physician
```

---

## Stack

| Component | Technology |
|---|---|
| Vector database | Actian VectorAI DB (Docker, port 50051) |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, local cache) |
| Search type | Hybrid: semantic + filtered by region and age group |
| Backend | Python + Flask |
| Interface | HTML/CSS/JS — mobile-first, no frameworks |
| Local storage | JSON files → sync on reconnection |

---

## How to run

**Requirements:** Docker Desktop, Python 3.10+

```bash
# 1. Start the database
git clone https://github.com/hackmamba-io/actian-vectorAI-db-beta
cd actian-vectorAI-db-beta
docker compose up -d

# 2. Set up the project
cd ../pulse
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install ../actian-vectorAI-db-beta/actian_vectorai-0.1.0b2-py3-none-any.whl
pip install flask sentence-transformers

# 3. Download the embedding model (one time only — works offline after)
python download_modelo.py

# 4. Populate the clinical knowledge base
python seed_database.py

# 5. Run
python app.py
```

Open `http://localhost:5000` on any device on the same Wi-Fi network.

---

## Repository

[github.com/Nago13/pulse-acs](https://github.com/Nago13/pulse-acs)

---

*Built during the Actian VectorAI DB Build Challenge — April 2026.*

*We didn't build Pulse for the billion people who already have AI.*
*We built it for the places everyone else forgot.*
