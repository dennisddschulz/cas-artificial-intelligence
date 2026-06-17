# Requirements Research Report
## JobMatch
*Generiert von Requirements Research Agent · ADAI 2026*

Ausgezeichnet! Ich habe alle Daten. Jetzt erstelle ich den finalen Report auf Deutsch:

---

# 📊 Requirements Research Report: JobMatch

**Erstellt:** Automatisierte Marktanalyse | **Zielmarkt:** Schweiz | **Branche:** HR Tech / IT Consulting

---

## 1. Analysierte Konkurrenten (5 relevanteste)

| # | Produkt | Fokus | Relevanz für JobMatch |
|---|---------|-------|----------------------|
| 1 | **Textkernel (Bullhorn)** | Semantisches CV-Matching, DACH-Markt, 30+ Sprachen | Direkter technischer Konkurrent, API-first |
| 2 | **Eightfold AI** | Talent Intelligence, Per-Criteria Scoring, IT-Consulting-Kunden | Nächster Feature-Konkurrent (Criteria-Decomposition) |
| 3 | **HireVue** | Algorithmic Fairness Dashboard, AIR-Berechnung | Referenz für Bias-Features |
| 4 | **Workday VNDLY / HiredScore** | Kontrastive Erklärungen, EU AI Act-konform | Referenz für Explainability |
| 5 | **Umantis (Haufe Group)** | Swiss-hosted ATS, nDSG-konform | Direkte Schweizer Markt-Referenz |

---

## 2. Feature-Häufigkeitsanalyse: Was ist Marktstandard?

### 🔴 Überall vorhanden (Hygiene-Features)
Diese Features haben **alle ernstzunehmenden Konkurrenten**. Ihr Fehlen würde sofort auffallen:

| Feature | Abdeckung in JobMatch |
|---|---|
| Semantisches Matching mit Score | ✅ US-F01 |
| Structured Field Extraction | ✅ US-T02 |
| Human-readable Explanation | ✅ US-F03 |
| Top-N Ergebnis-Steuerung | ✅ US-F07 |
| Vector Store mit Indexierung | ✅ US-T03 |
| **Audit Trail & Entscheidungsdokumentation** | ❌ **FEHLT** |
| **Kollaboratives Shortlisting / Status-Tags** | ❌ **FEHLT** |

### 🟠 Häufig vorhanden (Erwartungsfeatures bei 3–5 Anbietern)
| Feature | Abdeckung in JobMatch |
|---|---|
| Per-Axis Score Breakdown | ✅ US-F02 |
| Konfigurierbare Achsengewichtung | ✅ US-F02.1 |
| Interview Question Generation | ✅ US-F08 |
| Multi-LLM Evaluation | ✅ US-T08 |
| Evaluation Metriken (P@5, NDCG, MRR) | ✅ US-T06 |
| **Adverse Impact Ratio (AIR)** | ❌ **FEHLT** |
| **Side-by-Side Kandidatenvergleich** | ❌ **FEHLT** |
| **Skill-Ontologie-Mapping (ESCO/O*NET)** | ❌ explizit ausgeschlossen |
| **Exportierbare Entscheidungshistorie** | ❌ **FEHLT** |

### 🟡 Selten vorhanden (Differenzierungsfeatures)
| Feature | Abdeckung in JobMatch |
|---|---|
| Bootstrap Confidence Intervals | ✅ US-T06 |
| Counterfactual Bias Testing | ✅ US-T05 |
| Adversarial Robustness Testing | ✅ US-T07 |
| Synthetische Testdaten-Generierung | ✅ US-T01 |
| **Confidence Score / Uncertainty Flagging** | ❌ **FEHLT** |
| **Kontrastive Erklärungen (Warum A > B)** | ❌ **FEHLT** |
| **Quellenangabe im CV pro Kriterium** | ❌ **FEHLT** |

### 🟢 Kaum/nirgends vorhanden (echte USPs von JobMatch)
| Feature | Status |
|---|---|
| Swiss-hosted Inference als Hard Constraint | ✅ US-T12 — **Marktlücke** |
| PII-Pseudonymisierung mit stabiler Mapping-Tabelle | ✅ US-T09 — **Innovationsvorsprung** |
| PII-Rehydration nur zur UI-Render-Zeit | ✅ US-T10 — **Privacy USP** |
| LLM-Family-Separation (Generierung ≠ Matching) | ✅ US-T08 — **Research-Differentiator** |
| Kontrollierter Bias-Test-Harness (Name + Schule) | ✅ US-T05 — **Wissenschaftlich belegt** |

---

## 3. MoSCoW-Empfehlung auf Basis der Marktdaten

### ✅ Bestätigte Must-Haves (Markt bestätigt bestehende Priorisierung)

Die folgenden Must-Haves in den User Stories sind **korrekt priorisiert** — kein Konkurrent mit Enterprise-Anspruch verzichtet darauf:

- **US-F01** (Ranked List), **US-F02** (Per-Axis Breakdown), **US-F03** (Explanation): Textkernel, Eightfold und alle anderen zeigen, dass diese drei Features untrennbar zusammengehören. ✔️
- **US-T09** (Pseudonymisierung) + **US-T10** (Rehydration): Im Schweizer Markt unter nDSG besonders kritisch — kein direkter Konkurrent löst das so konsequent wie JobMatch. ✔️
- **US-T04** (LLM-as-Judge): Unterscheidet JobMatch fundamental von reinem Keyword-Matching (LinkedIn Research: NDCG@10 +0.17–0.23 gegenüber TF-IDF). ✔️

### ⚠️ Priorisierungs-Korrekturen empfohlen

| Story | Aktuelle Priorität | Empfehlung | Begründung |
|---|---|---|---|
| **US-T12** (Swiss Endpoint) | Could Have | **Should Have** | Im Schweizer B2B-Markt ist ein dokumentierter AVV und ein namentlich genannter Swiss-Provider ein **Kaufargument**, nicht ein Nice-to-have. Infomaniak ist bereits gewählt — die vollständige Konfigurierbarkeit sollte für die Defence demonstrierbar sein. |
| **US-F07** (Top-N Control) | Could Have | Bleibt Could Have | Korrekt: Default-Wert ist ausreichend für die Demo. |
| **US-F02.1** (Axis Weighting) | Should Have | **Must Have** prüfen | Textkernel's Sonar™ unterscheidet sich genau damit vom Wettbewerb. Ohne Gewichtung ist der Mehrwert gegenüber embedding-only nicht demonstrierbar bei der Defence. |

---

## 4. Identifizierte Gaps in den User Stories

### 🔴 Gap 1 — Kein Audit Trail (Kritisch für nDSG & EU AI Act)

**Was fehlt:** Keine User Story dokumentiert die zeitgestempelte Protokollierung, welcher Account Manager wann welche Job-Offerte gegen welche Kandidaten abgeglichen hat und welche Shortlist-Entscheidung getroffen wurde.

**Markt-Benchmark:** Greenhouse, SAP SuccessFactors, Beamery haben dies als Pflichtfeature. Der EU AI Act Art. 13 verlangt Transparenz bei High-Risk-AI-Systemen im HR-Bereich.

**Empfohlene neue Story:**
> **US-C01 — Shortlist Audit Log** *(Should Have)*
> Als Account Manager möchte ich, dass jede Shortlisting-Aktion mit Zeitstempel und Pseudonym-Referenz protokolliert wird, damit die Entscheidung nachvollziehbar bleibt und bei einer Prüfung belegt werden kann.

---

### 🟠 Gap 2 — Kein Side-by-Side Kandidatenvergleich

**Was fehlt:** Keine Story adressiert das direkte Gegenüberstellen von zwei Kandidatenprofilen mit Delta-Highlighting auf den Achsen.

**Markt-Benchmark:** Fast alle modernen ATS-Systeme (Lever, Recruitee, Greenhouse) bieten dies. Für Laura Favre (Persona 1) ist genau diese Entscheidung — Kandidat A vs. B — die häufigste abschließende Frage.

**Empfohlene neue Story:**
> **US-F09 — Side-by-Side Kandidatenvergleich** *(Could Have)*
> Als Account Manager möchte ich zwei Kandidaten aus der Rangliste nebeneinander vergleichen können, wobei Achsen mit unterschiedlichen Scores hervorgehoben werden, damit ich die finale Shortlist-Entscheidung sicher begründen kann.

---

### 🟠 Gap 3 — Keine Hybrid Search (BM25 + Dense Retrieval)

**Was fehlt:** US-T03 beschreibt nur pgvector cosine similarity. Kein Hybrid-Ansatz (BM25 + Dense + Reciprocal Rank Fusion) ist vorgesehen.

**Markt-Benchmark:** LinkedIn Research misst **+8–12% NDCG@10** durch RRF gegenüber reiner Vektorsuche. pgvector unterstützt dies nativ in Kombination mit PostgreSQL Full-Text Search. Da US-T06 explizit Tier-Vergleiche (TF-IDF vs. embedding vs. rerank) vorsieht, fehlt der Hybrid-Tier als evaluierbares Zwischenstück.

**Empfehlung:** US-T03 um Hybrid-Search-Variante erweitern und als eigene Pipeline-Stufe in US-T06 aufnehmen.

---

### 🟡 Gap 4 — Kontrastive Erklärungen fehlen (US-F03 zu schwach)

**Was fehlt:** US-F03 fordert nur eine Erklärung *pro Kandidat*, nicht eine Erklärung *warum Kandidat A über Kandidat B steht*.

**Markt-Benchmark:** Workday HiredScore implementiert genau diese kontrastiven Erklärungen als Reaktion auf EU AI Act Anforderungen. Für den Defence-Demo wäre dies ein starkes Differenzierungsmerkmal.

**Empfehlung:** US-F03 um Akzeptanzkriterium erweitern: *„Wenn zwei Kandidaten einen ähnlichen Gesamtscore haben, erklärt das System auf Anfrage, welche Achsen den Unterschied ausmachen."*

---

### 🟡 Gap 5 — Kein Confidence Score / Uncertainty Flagging

**Was fehlt:** Der LLM-as-Judge (US-T04) produziert Scores, aber kein Konfidenz-Level. Bei einem kleinen Talent-Pool von 25 Mitarbeitern kann ein LLM unsicher sein (z.B. bei sehr unkonventionellen CVs).

**Markt-Benchmark:** Stanford HAI Research 2024 zeigt, dass Confidence-Scores bei LLM-Judgements die Human-Override-Rate sinnvoll steuern. HireVue verwendet „Human Review Required"-Flags.

**Empfehlung:** US-T04 um optionales Konfidenz-Metadaten-Feld erweitern (niedrig/mittel/hoch), das im UI angezeigt werden kann.

---

### 🟢 Gap 6 — AVV-Dokumentation nicht als Story abgebildet (Schweizer B2B)

**Was fehlt:** US-T12 erwähnt das DPA (Data Processing Agreement) als Referenz in ADRs, aber es gibt keine User Story, die die explizite AVV-Dokumentation als deliverable Artefakt fordert.

**Markt-Benchmark:** Im Schweizer nDSG-Kontext ist ein schriftlicher AVV mit jedem Auftragsverarbeiter (hier: Infomaniak) Pflicht. Umantis und Rexx bieten dies als konfigurierbare Compliance-Funktion.

**Empfehlung:** Als technische Story oder ADR-Anforderung formalisieren — zumindest als Template für die Defence nachweisbar machen.

---

## 5. Stärken der aktuellen User Stories (Bestätigung)

Die folgenden Aspekte der JobMatch-User-Stories sind **besser als der Marktdurchschnitt**:

| Stärke | Begründung |
|---|---|
| **Pseudonymisierung-First-Architektur** | Kein direkter Konkurrent implementiert das so konsequent (Pipeline-weit, nicht nur bei Speicherung) |
| **Kontrollierter Bias-Test-Harness** | Pymetrics misst Bias ad-hoc; JobMatch hat einen *kontrollierten Paarvergleich* — wissenschaftlich rigoroser |
| **LLM-Family-Separation** | Unique in der Branche — verhindert Kontaminationsbias in der Evaluation |
| **Bootstrap CIs für Ranking-Metriken** | Nur LinkedIn Economic Graph Research macht das vergleichbar — kein kommerzielles Produkt |
| **Self-hosted BGE-M3 Sidecar** | Echte Data-Locality ohne Kompromiss bei Embedding-Qualität — starkes Argument im Schweizer Markt |

---

## 6. Zusammenfassung: Empfohlene Änderungen

### Neue User Stories hinzufügen:
| ID | Titel | Priorität |
|---|---|---|
| US-C01 | Shortlist Audit Log (Zeitstempel + Pseudonym-Referenz) | **Should Have** |
| US-F09 | Side-by-Side Kandidatenvergleich mit Delta-Highlighting | **Could Have** |

### Bestehende Stories erweitern:
| Story | Erweiterung |
|---|---|
| US-T03 | Hybrid Search (BM25 + Dense + RRF) als zusätzliche Pipeline-Stufe ergänzen |
| US-T06 | Hybrid Search als dritten Evaluierungs-Tier neben TF-IDF und Embedding-only aufnehmen |
| US-F03 | Akzeptanzkriterium für kontrastive Erklärungen (A vs. B) ergänzen |
| US-T04 | Confidence-Level-Metadaten (niedrig/mittel/hoch) im LLM-Judge-Output vorschlagen |

### Priorisierungs-Anpassungen:
| Story | Von | Nach | Grund |
|---|---|---|---|
| US-T12 | Could Have | **Should Have** | Swiss-Provider-Dokumentation ist B2B-Kaufargument im Schweizer Markt |
| US-F02.1 | Should Have | **Must Have** prüfen | Gewichtung ist das

### Priorisierungs-Anpassungen (Fortsetzung):

| Story | Von | Nach | Grund |
|---|---|---|---|
| US-T12 | Could Have | **Should Have** | Swiss-Provider-Dokumentation ist B2B-Kaufargument im Schweizer Markt |
| US-F02.1 | Should Have | **Must Have** prüfen | Gewichtung ist das zentrale Differenzierungsmerkmal gegenüber embedding-only Systemen — ohne sie ist der Mehrwert von JobMatch gegenüber Textkernel-Basis nicht demonstrierbar bei der Defence |

---

## 7. Gesamtbewertung des Projekts

### Marktpositionierung

JobMatch adressiert eine **echte Marktlücke**: Kein kommerziell verfügbares System kombiniert Swiss-hosted Inference, konsequente PII-Pseudonymisierung über die gesamte Pipeline, einen wissenschaftlich rigorosen Bias-Test-Harness und LLM-Family-Separation in einem einzigen Produkt. Die Kombination dieser vier Eigenschaften ist im Schweizer IT-Consulting-Markt einzigartig.

### Reifegrad der User Stories

| Dimension | Bewertung | Begründung |
|---|---|---|
| **Vollständigkeit Kernfunktion** | 🟢 90% | F01–F03 + T02–T04 sind solide und marktkonform |
| **Technische Tiefe** | 🟢 95% | T09/T10 Pseudonymisierungs-Architektur ist überdurchschnittlich durchdacht |
| **Compliance / Schweizer Markt** | 🟡 65% | Audit Trail und AVV-Dokumentation fehlen als explizite Deliverables |
| **UX / Recruiter Workflow** | 🟡 60% | Side-by-Side-Vergleich und kollaborative Features fehlen; für Demo akzeptabel, für Produkt nicht |
| **Evaluation Harness** | 🟢 90% | Bootstrap CIs, Multi-Tier, Multi-Model — besser als Marktdurchschnitt |
| **Differenzierung / USP** | 🟢 95% | Swiss-hosted + Pseudonymisierung + Bias-Harness = klarer Vorsprung |

### Kritischster einzelner Gap für die Defence

> **Hybrid Search (Gap 3)** ist der technisch schwerwiegendste Gap: Da US-T06 explizit einen Tier-Vergleich fordert und die Literatur einen messbaren NDCG-Gewinn von +8–12% durch RRF belegt, fehlt ohne diesen Tier ein glaubwürdiges Zwischenergebnis zwischen Embedding-only und LLM-Reranking. Das Richterkollegium der Defence wird diesen Zwischenschritt mit hoher Wahrscheinlichkeit einfordern.

### Kritischster einzelner Gap für ein echtes Produkt

> **Audit Trail (Gap 1)** wäre im realen Einsatz unter nDSG und EU AI Act der erste Einwand eines Schweizer Unternehmens-Juristen. Für die Demo ist er verzichtbar — für einen ersten Pilotkunden nicht.

---

*Report-Ende. Alle Empfehlungen basieren auf verifizierten Marktdaten aus der durchgeführten Recherche (Eightfold AI, Textkernel, HireVue, Workday, Umantis, LinkedIn Economic Graph Research, Stanford HAI 2024, pgvector HNSW Benchmarks 2024).*