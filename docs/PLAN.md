# Thèse MSc — Autonomous Trading Agents via Model Context Protocol
### Plan d'action 2 semaines · Architecture · Outils · Prompts Claude Code

> Superviseur : Cristopher Salvi (Imperial). Sujet 1.22.
> Hypothèses retenues (à confirmer) : priorité **rigueur d'évaluation** + multi-agent modéré ;
> actifs **actions US** d'abord ; LangGraph supposé nouveau pour toi.

---

## 0. La thèse en une phrase, et le piège à éviter

Le sujet de Salvi te demande trois choses imbriquées :
1. **Une architecture** où un agent LLM accède à market data / indicateurs / sentiment / exécution **via des serveurs MCP** ;
2. **Un backtest rigoureux** comparé à des baselines systématiques (momentum, mean-reversion) ;
3. **Une évaluation multi-dimensionnelle** : latence, rendement ajusté du risque, **et qualité du raisonnement (CoT) selon les régimes de marché**.

Le piège — documenté noir sur blanc dans le papier **KellyBench** que Salvi t'a partagé — est le *knowledge-action gap* : les modèles articulent de belles stratégies mais (a) introduisent du look-ahead/leakage dans leur backtest, (b) ne s'adaptent pas aux changements de régime, (c) n'exécutent pas ce qu'ils raisonnent. **Ta valeur ajoutée de thèse, c'est de construire le harnais qui mesure proprement ce gap pour des agents MCP.** Le jury notera la rigueur méthodologique, pas le nombre d'agents.

**Positionnement vs l'état de l'art (voir §1)** : TradingAgents existe déjà et fait le multi-agent "fonds". Si tu te contentes de le refaire, c'est faible. Ton angle différenciant = **MCP comme couche d'outils standardisée + protocole d'évaluation du CoT par régime**. C'est neuf et c'est exactement ce que demande l'intitulé.

---

## 1. État de l'art (synthèse de la recherche)

### 1.1 Frameworks multi-agent trading — repos sur lesquels CONSTRUIRE

| Repo / Papier | Ce que c'est | Comment l'utiliser |
|---|---|---|
| **TradingAgents** (TauricResearch, arXiv 2412.20138) | LE framework de référence. Rôles : analystes fondamental/sentiment/technique, chercheurs Bull/Bear en débat, trader, équipe risk. v0.2 (2026) multi-provider. | **Base à forker/étudier.** Récupère la structure de rôles et le pattern de débat Bull/Bear. NE PAS tout reprendre. |
| **HedgeAgents** (WWW '25, arXiv 2502.13165) | Fund manager central + experts par classe d'actifs, coordination par "conférences". | Inspiration pour la hiérarchie manager↔experts si tu veux la couche "fonds". |
| **FinCon** | Hiérarchie manager-analyste, comm. en langage naturel, contrôle du risque. | Pattern de communication hiérarchique. |
| **chmbrs/hedge_fund_agents** | Fork local-first (Ollama) de TradingAgents, plus narrow. | Exemple de simplification — utile pour voir quoi couper. |

### 1.2 Serveurs MCP financiers — N'EN ÉCRIS PAS DEPUIS ZÉRO

L'écosystème MCP finance est déjà riche. Réutilise au maximum :

| Serveur MCP | Couverture |
|---|---|
| **financial-datasets/mcp-server** | Income statements, balance sheets, cash flow, prix, news. Très propre, officiel-like. |
| **twelvedata/mcp** | Streaming temps réel WebSocket, time series, quotes, forex/crypto. "u-tool" = routeur NL. |
| **cfdude/mcp-finnhub** | Quotes, candles, 50+ indicateurs techniques, news+sentiment, fundamentals, SEC filings. |
| **fintools-ai/mcp-market-data-server** | Volume profile, insights structurés pour agents. |
| Le `trading_server.py` du TP gen-ai (que tu as déjà) | Ton point de départ minimal : 9 tools FastMCP déjà écrits. |

> **Décision archi clé** : ton propre serveur MCP n'expose QUE ce qui est spécifique à ta thèse (exécution paper-trading, backtest, indicateurs custom). Pour la data, tu **branches les serveurs MCP existants**. C'est exactement le propos du sujet : l'agent compose des serveurs MCP hétérogènes.

### 1.3 Benchmarks d'évaluation LLM trading — TON CADRE MÉTHODOLOGIQUE

Lis ces papiers, ils te donnent les métriques et le vocabulaire que le jury attend :

- **StockBench** (Chen et al., oct. 2025) — benchmark *contamination-free*, multi-mois, décisions buy/sell/hold séquentielles. **Le plus proche de ton setup.** Reprends sa méthodo "contamination-free".
- **LiveTradeBench** (Yu et al., nov. 2025) — temps réel, 21 LLMs, lookback 10 jours pour éviter le leakage. Résultat clé à citer : **les scores LLM généralistes (LMArena) ne corrèlent PAS avec la perf trading** (ρ≈0). Argument fort pour ta thèse.
- **Agent Market Arena (AMA)** (arXiv 2510.11695) — compare InvestorAgent/TradeAgent/HedgeFundAgent/DeepFundAgent × backbones. Montre que **l'architecture agent compte plus que le backbone**. Justifie ton choix de varier l'archi.
- **ATLAS** (arXiv 2510.15949) — prompt optimization + coordination multi-agent, surtout en **régimes volatils**. Pertinent pour ton axe "régimes".
- **KellyBench** (le PDF de Salvi) — ton **garde-fou méthodologique**. Failles cataloguées : non-stationarité (équipes promues = changement de régime), Kelly codé mais jamais appelé, leakage via `CalibratedClassifierCV` sur tout le train set, déclarations prématurées de fin. **Cite-le pour montrer que tu connais les pièges et que ton design les évite.**

### 1.4 Backtesting — éviter le look-ahead bias par l'architecture

Consensus net de la recherche : le look-ahead bias est *l'erreur cardinale* et la seule défense robuste est **architecturale**, pas la discipline du dev.

- **Event-driven > vectorisé** pour le réalisme et l'absence de leakage (drip-feed bar par bar). Réf : QuantStart event-driven series.
- **vectorbt** : ultra-rapide (Numba), idéal pour le *grid search* de baselines et le tuning de seuils. Risque de leakage si mal utilisé.
- **backtrader** : event-driven natif, broker simulé (commissions, slippage, types d'ordres). Plus lent mais sûr.
- **Recommandation** : `vectorbt` pour explorer les baselines vite + un **moteur event-driven** (le tien, simple, ou backtrader) pour la boucle agent où le réalisme prime. Ton `backtest.py` actuel est vectorisé et léger — OK pour démarrer, mais documente ses limites.

---

## 2. Architecture cible

### 2.1 Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATEUR (LangGraph)                    │
│                  graphe d'état, checkpoints, HIL                 │
│                                                                  │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐   │
│   │ Analyste │   │ Analyste │   │ Analyste │   │  Risk /     │   │
│   │ Technique│   │ Sentiment│   │ Fondament│   │  Portfolio  │   │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬──────┘   │
│        │              │              │                │          │
│        └──────────────┴───── débat ──┴────────────────┘          │
│                          │                                       │
│                    ┌─────▼──────┐                                │
│                    │   Trader   │  ← décision finale + CoT loggé │
│                    └─────┬──────┘                                │
└──────────────────────────┼──────────────────────────────────────┘
                           │  appels d'outils via MCP (stdio/HTTP)
        ┌──────────────────┼──────────────────────┐
        ▼                  ▼                       ▼
┌───────────────┐  ┌────────────────┐     ┌──────────────────┐
│ MCP data       │  │ MCP "thesis"   │     │ MCP exécution     │
│ (existants):   │  │ (le tien):     │     │ (le tien):        │
│ financial-     │  │ • indicateurs  │     │ • place_order     │
│ datasets,      │  │   custom       │     │ • portfolio       │
│ twelvedata,    │  │ • régime detect│     │ • pnl             │
│ finnhub        │  │ • backtest     │     │ (paper, in-mem)   │
└───────────────┘  └────────────────┘     └──────────────────┘
        │
        ▼  (la même couche MCP sert le LIVE et le BACKTEST)
┌──────────────────────────────────────────────────────────────┐
│  TIME-MACHINE / REPLAY : un "clock" injecte les barres une par │
│  une. Les serveurs MCP ne renvoient QUE les données ≤ t_now.   │
│  → look-ahead structurellement impossible.                     │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Le concept central : "MCP-as-time-machine"

C'est **ta contribution architecturale la plus élégante** et elle tue le look-ahead bias à la racine :

- Tu introduis une horloge de simulation `t_now`.
- Tes serveurs MCP (et des wrappers autour des serveurs externes) ne retournent JAMAIS de données postérieures à `t_now`.
- L'agent appelle exactement les **mêmes outils MCP** en backtest et en live ; seule l'horloge change. C'est *le* point qui rend ton MCP-layer scientifiquement intéressant : la standardisation MCP permet une équivalence backtest/live parfaite.
- Pour le sentiment/news : pré-télécharge un corpus horodaté (avec **Firecrawl**, via tes crédits YC) et sers-le filtré par date. Évite le leakage temporel que LiveTradeBench gère avec un lookback fixe.

### 2.3 Choix d'orchestration : LangGraph

- **Pourquoi LangGraph** : stateful, graphe explicite (chaque étape de raisonnement = nœud traçable), checkpoints, human-in-the-loop, et surtout **observabilité native via LangSmith** (que tu as via tes crédits YC — parfait pour logguer et *évaluer le CoT*). C'est le standard production 2026 et c'est exactement l'outil pour ton axe "qualité du raisonnement".
- **Alternative plus simple** si LangGraph te ralentit : commence par un orchestrateur Python custom (boucle perceive→reason→act→observe). Beaucoup d'équipes tournent en custom. Mais tu perds l'observabilité gratuite.
- **NE PAS** partir sur CrewAard/AutoGen : moins adaptés à ton besoin de contrôle fin de l'état et du logging.

### 2.4 Stack technique recommandée

| Couche | Choix | Pourquoi |
|---|---|---|
| Orchestration | **LangGraph** (+ LangSmith pour traces) | Stateful, observable, HIL, MCP-compatible |
| Protocole outils | **MCP** (FastMCP côté serveurs) | C'est le sujet. stdio en local, HTTP si AWS |
| Data MCP | financial-datasets + finnhub + twelvedata | Réutilise l'existant |
| LLM backbones | **GPT-4.1/5.x via tes $2500 API OpenAI** (agents en prod) + Claude via abo Pro (dev/archi dans Claude Code) | Sépare le budget : OpenAI fait tourner les expériences, Claude Code écrit le code |
| Backtest | vectorbt (baselines) + moteur event-driven (boucle agent) | Vitesse + réalisme |
| Indicateurs | ton `indicators.py` + `pandas-ta` | Déjà écrit |
| Sentiment | FinBERT (ton `sentiment.py`) + option LLM | Déjà écrit |
| Scraping news horodaté | **Firecrawl** (crédits YC) | Corpus reproductible sans leakage |
| Détection de régime | hand-rolled (vol realisée, trend) + option HMM | Axe "régimes" du sujet |
| Compute | local d'abord ; **AWS ($10k si validé)** pour les runs longs/parallèles | Économise l'API en debug local |
| Versioning expériences | git + un simple `runs/` horodaté (ou MLflow si tu veux) | Reproductibilité |

> **Note budget LLM** : ton abo Claude **Pro** (~$20/mois) donne accès à Claude Code mais avec des limites (fenêtre 5h ~44k tokens, Opus très limité sur Pro — tu coderas surtout en Sonnet). Pour faire *tourner les agents de trading en masse*, utilise tes **$2500 de crédits OpenAI**, pas l'abo Pro. Garde Claude Code pour **écrire et refactorer le code**. Si tu prévois des sessions Claude Code intensives sur 2 semaines, envisage Max 5x temporairement — mais teste Pro d'abord, tu n'es pas obligé.

---

## 3. Métriques d'évaluation (le cœur de la note)

### 3.1 Performance financière (vs baselines)
- Rendement cumulé, **Sharpe**, **Sortino**, **Calmar**, max drawdown, turnover, hit rate.
- **Baselines obligatoires** (le sujet les nomme) : Buy & Hold, **momentum** (time-series + cross-sectional), **mean-reversion** (Bollinger/RSI). Optionnel : un ML forecaster standalone (LightGBM sur features techniques).
- Test de significativité : pas juste un chiffre. Bootstrap des rendements quotidiens (KellyBench le fait — inspire-toi), p-values, error bars. Le single-run ne prouve rien.

### 3.2 Latence / coût (axe "systèmes")
- Latence par décision (perceive→act), nb de tool calls, tokens consommés, coût $ par décision. AMA/KellyBench rapportent les tool calls — fais pareil.

### 3.3 Qualité du raisonnement (CoT) — TON DIFFÉRENCIATEUR
C'est là que tu peux innover et c'est explicitement dans l'intitulé. Pistes :
- **Faithfulness** : le trade exécuté correspond-il au raisonnement ? (mesure directe du *knowledge-action gap* de KellyBench). Tu peux scorer automatiquement : parse la décision raisonnée vs l'ordre réel.
- **Grounding** : le raisonnement cite-t-il les outputs d'outils réellement appelés, ou hallucine-t-il des chiffres ? (LangSmith te donne les vrais outputs).
- **Rubric de sophistication** façon KellyBench (ils ont un barème de 52 points) : construis un mini-rubric (gestion du risque, prise en compte de l'incertitude, adaptation au régime) et fais noter par un LLM-juge + toi-même sur un échantillon.
- **Sensibilité au régime** : segmente toutes les métriques ci-dessus par régime (bull/bear/range/high-vol). Le résultat attendu et intéressant : le CoT se dégrade en haute volatilité (cf. ATLAS, LiveTradeBench).

### 3.4 Régimes de marché
- Définis 3-4 régimes sur ta période de test (ex. 2022 bear, 2023-24 bull, épisodes high-vol).
- Méthode simple défendable : volatilité réalisée glissante + pente de tendance → buckets. Mentionne HMM comme extension.
- **Rapporte toutes les métriques par régime.** C'est ce qui transforme un projet "j'ai fait trader un LLM" en "j'ai caractérisé QUAND et POURQUOI ça marche ou pas".

---

## 3b. Echelles de run (tiers)

Trois niveaux documentés dans `backtest/tier.py`. Chaque run doit indiquer son tier dans
les rapports : les résultats sont étiquetés "smoke-scale", "medium-scale", etc.

| Tier | Dates max | Tickers max | Warmup requis | Coût-ack | Usage |
|------|-----------|-------------|---------------|----------|-------|
| **smoke** | 5 | 3 | REGIME_WARMUP_CALENDAR_DAYS (420j) | non | Dev, CI, smoke tests |
| **medium** | 22 (~1 mois) | 3 | idem | oui | Première validation réelle par régime |
| **full** | illimité | 10 | idem | oui | Run final unique de thèse |

**Règle warmup obligatoire pour tous les tiers :** la fenêtre de DONNÉES = fenêtre décision +
`REGIME_WARMUP_CALENDAR_DAYS` (420j) pour garantir `n_warmup_excluded=0` dès la 1ère
décision. Les deux engines importent `REGIME_WARMUP_CALENDAR_DAYS` depuis `regime.py`.

**Fenêtre medium recommandée :** AAPL+MSFT+NVDA, 2023-01-03→2023-01-31 (19 trading days,
bear→bull→range transition). Justification : Jan 2023 est le premier mois de récupération
après le crash 2022 — la 20d-trend passe de bear (début janvier) à bull (mi-janvier) pour
AAPL/MSFT. Données en cache depuis Run #4 → re-runs $0.

---

## 4. Planning jour par jour (2 semaines)

> Règle d'or : à la fin de **chaque** journée tu dois avoir un truc qui tourne. Pas de big bang d'intégration en J13.
> Principe KellyBench : ferme la boucle tôt, vérifie que ce qui est raisonné est exécuté.

### Semaine 1 — Fondations + boucle agent qui tourne

**J1 — Cadrage & état de l'art actif**
- Lis (en diagonale ciblée) : TradingAgents (archi), StockBench (méthodo éval), KellyBench (pièges). 2-3h max.
- Clone TradingAgents + 1-2 serveurs MCP finance. Fais-les tourner en local. Objectif : *voir* le pattern, pas tout comprendre.
- Écris un `README` de thèse : question de recherche, hypothèses, métriques (copie §3). Ce doc devient ton fil rouge et le squelette de rédaction.

**J2 — Couche MCP data + horloge**
- Mets en place ton serveur MCP "thesis" (pars du `trading_server.py` du TP).
- Implémente l'**horloge `t_now`** et le filtrage temporel (le truc le plus important de la semaine).
- Branche financial-datasets ou finnhub MCP. Smoke test : l'agent récupère un prix à une date passée sans voir le futur.

**J3 — Baselines + backtest engine**
- vectorbt : Buy&Hold, momentum, mean-reversion sur ton univers (5-10 tickers US liquides). Métriques §3.1.
- Ces baselines sont ton **étalon** : tu dois les avoir AVANT l'agent, sinon tu ne sauras pas si l'agent est bon.

**J4 — Agent single-role minimal**
- Un seul agent "trader" dans LangGraph qui : appelle les outils MCP, raisonne, place un ordre paper. Boucle perceive→reason→act→observe.
- Logge TOUT dans LangSmith (CoT, tool calls, latence).
- Fais-le tourner sur 1 mois de données. Vérifie l'absence de look-ahead (audit manuel de quelques décisions).

**J5 — Backtest agent complet + harnais d'éval v1**
- Fais tourner l'agent single-role sur toute la période de test via l'horloge.
- Code le calcul automatique des métriques §3.1 + §3.2.
- Premier tableau agent vs baselines. Même mauvais, c'est un résultat.

**J6-J7 — Multi-agent + tampon**
- Ajoute 2-3 rôles (technique, sentiment, risk) + un nœud "trader" qui synthétise. Pattern de débat optionnel.
- Garde le single-agent comme baseline d'archi (cf. AMA : compare les archis).
- Week-end = tampon pour rattraper le retard (il y en aura).

### Semaine 2 — Évaluation, régimes, rédaction

**J8 — Détection de régime**
- Implémente les buckets de régime. Tag chaque jour de la période de test.
- Re-segmente les métriques existantes par régime.

**J9 — Évaluation du CoT**
- Implémente faithfulness + grounding (parsing automatique depuis les traces LangSmith).
- Mini-rubric de sophistication + LLM-juge sur un échantillon de décisions.

**J10 — Expériences principales**
- Lance les runs finaux : {single-agent, multi-agent} × {2 backbones, ex. GPT-4.1 et GPT-5.x} × seeds multiples.
- Utilise AWS si validé pour paralléliser. Sinon, séquence sur la nuit.
- **Multi-seed obligatoire** (KellyBench montre que le single-seed ment).

**J11 — Analyse + significativité**
- Bootstrap, error bars, tests. Tableaux et figures par régime.
- Cherche LE résultat narratif : "l'agent bat les baselines en range mais s'effondre en high-vol et son CoT devient moins fidèle" (hypothèse plausible).

**J12-J13 — Rédaction**
- Méthodo (archi MCP, horloge, métriques), résultats, discussion. Réutilise tes figures.
- Section "pièges évités" en citant KellyBench → montre la maturité méthodologique.

**J14 — Buffer + handoff**
- Nettoyage repo, reproductibilité (un script qui relance tout), notes pour la suite (tu auras moins de temps après).
- Envoie un point d'avancement à Salvi.

---

## 5. Comment faire bosser l'IA au maximum (workflow Claude Code / Cursor)

### 5.1 Répartition des outils
- **Claude Code (abo Pro)** : écriture/refactor du code, mise en place de l'archi, debug. Travaille en **Sonnet** par défaut (Opus limité sur Pro). Utilise le **plan mode** pour les grosses tâches et `/compact` pour gérer le contexte sur longues sessions.
- **Cursor Pro+** : édition interactive fine, complétion, quand tu veux garder la main fichier par fichier. Bon complément à Claude Code.
- **API OpenAI ($2500)** : **faire tourner les agents de trading** (les expériences). C'est ton carburant de calcul, pas ton outil de code.
- **LangSmith (YC)** : observabilité/traces/éval CoT.
- **Firecrawl (YC)** : corpus news horodaté.
- **AWS ($10k si validé)** : runs longs et parallèles en semaine 2.

### 5.2 Méthode pour déléguer à Claude Code sans perdre le contrôle
1. **Un `CLAUDE.md` à la racine** du repo qui décrit l'archi, les conventions, les pièges (look-ahead !), et la définition de "fini". Claude Code le lit automatiquement.
2. **Travaille par tâches atomiques** correspondant aux journées du planning. Une tâche = un prompt clair + critère de done + test.
3. **Toujours demander des tests** : un module sans test qui "marche" est une dette. KellyBench montre que le code peut être correct mais jamais appelé — les tests d'intégration de la boucle sont critiques.
4. **Revue humaine des points sensibles** : l'horloge/anti-look-ahead et le calcul des métriques. Ne délègue pas la confiance là-dessus.

### 5.3 Prompt de démarrage pour Claude Code (à coller en J1-J2)

> Copie-colle ceci dans Claude Code, à la racine d'un repo vide. Adapte les `<...>`.

```
Tu es mon copilote d'ingénierie pour ma thèse de MSc à Imperial :
"Autonomous Trading Agents via Model Context Protocol". Lis ce brief en entier
avant d'écrire du code, puis propose un PLAN (pas de code) que je validerai.

CONTEXTE & OBJECTIF
- Construire un système où un/des agent(s) LLM accèdent à market data,
  indicateurs techniques, sentiment de news et exécution d'ordres UNIQUEMENT
  via des serveurs MCP, dans une boucle agentique perceive→reason→act→observe.
- Évaluer l'agent par backtest sur actions US, vs baselines systématiques
  (Buy&Hold, momentum, mean-reversion), selon 3 axes : rendement ajusté du
  risque, latence/coût, et QUALITÉ DU RAISONNEMENT (fidélité CoT↔action,
  grounding sur outputs d'outils, sophistication) — segmentés par RÉGIME de marché.

CONTRAINTE MÉTHODOLOGIQUE NON NÉGOCIABLE — ANTI LOOK-AHEAD
- Architecture "MCP-as-time-machine" : une horloge de simulation t_now ;
  les serveurs MCP ne renvoient JAMAIS de données postérieures à t_now.
- L'agent doit appeler EXACTEMENT les mêmes outils MCP en backtest et en live ;
  seule l'horloge diffère. Le look-ahead doit être structurellement impossible,
  pas évité par discipline. Écris des tests qui PROUVENT qu'aucune donnée future
  ne fuite (ex. un test qui échoue si un outil renvoie une barre datée > t_now).

STACK IMPOSÉE
- Python 3.11+, MCP via FastMCP, orchestration LangGraph (+ traces LangSmith).
- Backtest : vectorbt pour les baselines ; moteur event-driven (drip-feed
  bar par bar) pour la boucle agent.
- Data : réutiliser des serveurs MCP existants (financial-datasets / finnhub)
  derrière des wrappers qui imposent le filtre temporel t_now.
- LLM backbone des agents = API OpenAI (je fournis la clé via variable d'env).
- Indicateurs et sentiment : je fournirai indicators.py et sentiment.py existants.

ARCHITECTURE CIBLE (à challenger si tu vois mieux)
- Serveur MCP "execution" : place_order, get_portfolio, get_pnl (paper, in-memory).
- Serveur MCP "analytics" : indicateurs custom, détection de régime, backtest.
- Wrappers MCP "data" autour des serveurs externes, avec filtre t_now.
- Orchestrateur LangGraph : commence par un SEUL agent trader ; prévois
  l'extension à analystes (technique/sentiment/risk) + trader synthétiseur.
- Logging structuré de chaque décision : inputs, tool calls, CoT, ordre, latence.

LIVRABLES DE CETTE PREMIÈRE ÉTAPE (et SEULEMENT celle-ci)
1. Un plan d'architecture détaillé + arborescence de fichiers.
2. Le squelette du repo : pyproject/requirements, CLAUDE.md, structure de modules,
   stubs typés avec docstrings, et la couche horloge + filtre temporel.
3. Les tests anti-look-ahead AVANT l'implémentation complète (TDD).
NE CODE PAS encore les agents ni les stratégies. Montre-moi le plan d'abord.

CONVENTIONS
- Type hints partout, docstrings, pas de magie cachée.
- Chaque module a des tests pytest. Tests d'intégration de la boucle = priorité.
- Commits atomiques avec messages clairs.
- Quand un choix a des trade-offs (ex. vectorbt vs event-driven), explique-les
  en 2 lignes avant de trancher.

Commence par me poser au maximum 3 questions de clarification si nécessaire,
puis donne le PLAN.
```

### 5.4 Prompts de suivi (exemples)
- *"Implémente maintenant l'horloge t_now et le wrapper data avec filtre temporel. Écris d'abord les tests qui prouvent l'absence de fuite future, puis le code. Montre-moi les tests qui passent."*
- *"Ajoute les 3 baselines (Buy&Hold, momentum TS, mean-reversion Bollinger) en vectorbt, avec calcul de Sharpe/Sortino/Calmar/maxDD. Sors un tableau comparatif sur <univers>/<période>."*
- *"Ajoute le calcul de faithfulness : parse la décision raisonnée depuis la trace et compare-la à l'ordre réellement passé. Rapporte le taux de divergence par régime."*

---

## 6. Risques & garde-fous

| Risque | Mitigation |
|---|---|
| Passer 2 semaines sur le plumbing multi-agent | Single-agent qui tourne dès J4. Multi-agent = bonus, pas prérequis. |
| Look-ahead bias non détecté | Tests automatiques anti-fuite (J2) + audit manuel. C'est non négociable. |
| Contamination weight-memory (le LLM "connaît" 2023-24) | Comme KellyBench : instruis l'agent de suivre une stratégie *rule-based* sur les outils ; choisis si possible une période récente/post-cutoff pour une partie des tests ; documente la limite honnêtement. |
| Single-seed trompeur | Multi-seed + bootstrap dès que possible. |
| Limites Claude Code Pro atteintes | Code en Sonnet, `/compact`, sessions ciblées ; bascule les *runs* sur API OpenAI ; Max 5x temporaire si vraiment bloqué. |
| Budget API brûlé en debug | Debug en local sur données cachées + petits modèles ; réserve GPT-5.x aux runs finaux. |
| Scope creep (FX, crypto, intraday…) | Actions US daily d'abord. Extensions = chapitre "future work". |

---

## 7. Décisions à confirmer avec moi / Salvi
1. **Univers & période** : 5-10 actions US liquides, période incluant ≥2 régimes (ex. 2022 bear → 2024 bull). OK ?
2. **Ampleur multi-agent** : je recommande 1 agent → puis 3-4 rôles max. Salvi veut-il explicitement la couche "fonds" complète (manager + experts), ou la rigueur d'éval prime-t-elle ?
3. **Contribution mise en avant** : "MCP-as-time-machine + éval du CoT par régime" comme thèse centrale — ça te convient comme angle ?
4. **Backbones** : GPT-4.1 + un GPT-5.x suffisent, ou Salvi veut-il un comparatif cross-provider (Claude/Gemini) ?

---

*Document de travail — à itérer. Prochaine étape suggérée : valider §7, puis lancer le prompt §5.3 dans Claude Code.*

---

## 8. Extensions futures (post-baseline, si temps disponible)

Ces deux extensions ont été identifiées comme intéressantes pour la thèse mais ne
sont **pas sur le chemin critique**.  Les implémenter seulement si le backtest PM
multi-agent est validé, les métriques d'évaluation sont complètes, et il reste au
moins 3-4 jours.

### 8.1 Agent "Quant Researcher"

**Idée** : ajouter un rôle d'agent qui peut lire la littérature financière (papers
dans `papers/`) et les résultats des backtests courants, puis **créer et valider ses
propres indicateurs techniques** (Ichimoku, custom ML features, etc.) et les mettre
à disposition de l'analyste technique via le serveur MCP `analytics/`.

**Architecture envisagée** :
- Le Quant Researcher a accès en lecture aux papers (résumés, `papers/README.md`),
  aux résultats de backtest (`results/*.csv`), et à un MCP `analytics/custom_tools`
  où il peut enregistrer de nouvelles fonctions indicateur.
- L'analyste technique peut découvrir et appeler ces indicateurs custom via MCP
  (`list_custom_indicators`, `compute_custom_indicator`).
- Le Quant Researcher tourne en mode "offline" (entre les runs, pas à chaque barre)
  pour amortir le coût LLM.

**Valeur pour la thèse** :
- Angle original : un agent qui étend lui-même la toolbox MCP est une contribution
  neuve par rapport à TradingAgents/HedgeAgents.
- Potentiellement mesurable : comparer les performances avec/sans les indicateurs
  custom du Quant Researcher.

**Risques** :
- Scope-creep majeur si mal délimité. Contraindre strictement : le Quant Researcher
  ne peut écrire que dans un sandbox Python vérifié, pas dans le code de prod.
- Sécurité : valider le code généré avant exécution (`ast.parse` + whitelist de
  bibliothèques). Ne jamais exécuter du code LLM non validé.

**Prérequis** : PM multi-agent validé + évaluation complète.

---

### 8.2 Fine-tuning par Reinforcement Learning (RL)

**Idée** : utiliser les décisions et retours réalisés enregistrés dans
`PMDecisionLog` (et `decisions.jsonl`) pour fine-tuner les agents via RL, par
exemple avec RLHF ou un signal de récompense basé sur le PnL/Sharpe incrémental.

**Pistes concrètes** :
- **PPO/GRPO sur les LLM** : utiliser OpenAI fine-tuning API (RLHF) ou une
  librairie open-source (TRL/veRL) pour entraîner un modèle à maximiser le
  Sharpe incrémental.  Signal de récompense = retour ajusté du risque à J+5.
- **Supervised fine-tuning (SFT) d'abord** : générer un dataset de paires
  (contexte marché → décision optimale rétrospectivement) depuis `decisions.jsonl`
  pour un premier SFT avant tout RL.
- **Évaluation hors-distribution** : crucial de tester sur une période out-of-sample
  (2024-2025) pour détecter le sur-apprentissage.

**Valeur pour la thèse** :
- Différenciateur fort si présenté proprement : "les agents s'améliorent par RL à
  partir de leur propre historique MCP".
- Aligné avec la tendance recherche (RLHF trading : cf. FinRL, AlphaPortfolio).

**Risques** :
- Complexité d'implémentation très élevée.  Ne pas commencer sans avoir 4+ jours
  disponibles et une infrastructure GPU.
- Overfitting sur le jeu de test existant si la frontière train/test n'est pas
  strictement respectée.
- Budget OpenAI fine-tuning potentiellement élevé.

**Prérequis** : backtest PM complet avec au moins 200+ décisions réelles loggées,
évaluation d'évaluation complète, période out-of-sample clairement définie.
