# LLM pod kapotou: Co vás ve škole nenaučili

Kniha, která vás provede světem moderních velkých jazykových modelů od základní teorie až po byznys nasazení a praktické projekty. Místo memorování suchých rovnic se zaměřuje na reálnou produkci a state-of-the-art (SOTA) technologie.

## Obsah

- **1.** Od dopředných sítí k sekvencím
- **2.** Transformer: Skutečný motor moderní AI
- **3.** Pre-training: Zrození gigantických modelů
- **4.** Alignment: Jak zkrotit umělou inteligenci
- **5.** Anatomie SOTA: MoE a multimodalita
- **6.** Magie pod kapotou: Optimalizace inference
- **7.** RAG: Propojení LLM s databázemi
- **8.** Agenti: Když modely používají nástroje
- **9.** Bezpečnost, evaluace a limity modelů
- **10.** Byznys nasazení a budoucí trendy
- **11.** Bonusový projekt I: Jak postavit vlastní Perplexity (Real-time Web Search Agent)
- **12.** Bonusový projekt II: Jak postavit Copilota ve VS Code (Coding Assistant s kontextem)
- **13.** Bonusový projekt III: Lokální AI pro pražský LegalTech startup (Privacy-First)

## Sestavení PDF

E-book se dynamicky generuje z jednotlivých částí `*.md` pomocí nástroje Pandoc a sázecího systému XeLaTeX.

Pro vygenerování `llms-book.pdf`:

1. Ujistěte se, že máte nainstalované: `python3`, a kompletní LaTeX distribuci (např. `texlive-full` / MacTeX).
2. Spusťte přiložený buildovací skript:
   ```bash
   ./scripts/generate-book-pdf.sh
   ```
   *Skript automaticky zkontroluje existenci `pypandoc` ve virtuálním prostředí (`.venv`), sloučí texty, upraví stylování a výsledné PDF uloží do rootu repozitáře. Dočasné buildovací soubory navíc zanechá bezpečně ignorované v ginu.*

