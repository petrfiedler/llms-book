# 1. Od dopředných sítí k sekvencím

## 1.1 Kde končí školní teorie: Proč dopředné sítě na text prostě nestačí
*Popis:* Intuitivní srovnání toho, co znáte ze školy (kde má vstup vždy přesně daný počet pixelů nebo hodnot), s reálným světem textu, kde má každá věta jinou délku. Ukážeme si, proč klasická "feed-forward" síť logicky zkolabuje, když se snaží zpracovat dynamickou sekvenci, a proč jsme museli změnit celou architekturu uvažování.

## 1.2 Překlad z lidštiny do matematiky: Tokeny a Embeddingy
*Popis:* Jak udělat ze slova (nebo jeho části) číslo, kterému síť rozumí. Představíme si embeddingy velmi vizuálně jako souřadnice na obrovské "mapě významů" (kde jsou král a královna blízko sebe). Jasně rozlišíme, co je **zastaralý koncept** (statický Word2Vec) a jak se to dělá dnes v **SOTA (State-of-the-Art) businessu** pomocí sub-word tokenizace (BPE) a kontextových embeddingů.

## 1.3 První pokus o čtení popořadě: Recurrent Neural Networks (RNN)
*Popis:* Ukážeme si základní princip sítě, která si předává "paměť" (hidden state) z jednoho slova na druhé – jako když čtete knihu. Tuto architekturu si probereme jen jako **důležitý historický koncept**, u kterého si rovnou řekneme, proč by vás dnes z každé AI firmy vyhodili, kdybyste na něm chtěli stavět jazykový model (problém mizejícího gradientu a katastrofální zapomínání u delších textů).

## 1.4 Záplata jménem LSTM (a GRU): Jak jsme chvíli opravovali neopravitelné
*Popis:* Vysvětlení Long Short-Term Memory na jednoduchém příkladu z běžného života (představte si vyhazovače v klubu, který pouští dovnitř jen důležité informace a ty zbytečné zapomíná). Upozorníme, že ačkoliv jde o **zastaralý koncept pro generování textu**, v businessu se s ním stále můžete občas setkat u specifických úloh (např. detekce anomálií v časových řadách z průmyslových senzorů). 

## 1.5 Úzké hrdlo sekvencí: Proč to všechno muselo jít do koše
*Popis:* Závěr kapitoly, který tvoří můstek k dnešním LLM. Vysvětlíme si největší slabinu celého přístupu RNN/LSTM: nutnost počítat slova *jedno po druhém*. Ukážeme si, proč je tento sekvenční přístup noční můrou pro moderní grafické karty (GPU) za miliony dolarů, které potřebují paralelizovat, a proč průmysl nutně potřeboval přijít s úplně novým řešením (Transformers), které dnes vládne světu.

# 2. Transformer: Skutečný motor moderní AI
Zde rozložíme architekturu Transformer na prvočinitele bez zbytečných matematických vzorců. Na jednoduchých příkladech si ukážeme mechanismus pozornosti (Self-Attention) a vysvětlíme, proč to byl historický zlom, který umožnil trénovat modely s nevídanou efektivitou.

# 3. Pre-training: Zrození gigantických modelů
Přesuneme se k fázi, kde se model učí "číst internet". Zaměříme se na to, jaká data se v byznysu reálně používají pro trénink, jak fungují obrovské GPU clustery a představíme zákony škálování (Scaling Laws), které definují dnešní vývoj obřích sítí.

# 4. Alignment: Jak zkrotit umělou inteligenci
Surový základní model je pro firemní nasazení nepoužitelný, protože umí jen doplňovat text. Zde čtenář pochopí, jak se pomocí Supervised Fine-Tuningu a RLHF (Reinforcement Learning from Human Feedback) z chaotického generátoru stane poslušný asistent.

# 5. Anatomie SOTA: MoE a multimodalita
Klíčová kapitola o tom, proč jsou modely jako Gemini nebo GPT-4 tak neuvěřitelně schopné a rychlé. Do detailu rozebereme architekturu Mixture of Experts (MoE) a ukážeme si nejnovější přístupy k tomu, jak dnešní systémy nativně vidí, slyší a chápou video.

# 6. Magie pod kapotou: Optimalizace inference
Tohle se ve škole neučí, ale pro praxi je to naprosto kritické. Vysvětlíme si, jak funguje KV cache pro plynulé generování textu a jak se dají obří modely zmenšit pomocí kvantizace tak, aby jejich provoz ve firmě nestál jmění.

# 7. RAG: Propojení LLM s databázemi
Absolutní byznysový standard a nutnost pro každého inženýra. Krok za krokem rozebereme Retrieval-Augmented Generation, vektorové databáze a strategie, jak model naučit chytře pracovat s vašimi interními dokumenty bez toho, aby si vymýšlel.

# 8. Agenti: Když modely používají nástroje
Posuneme se od pasivních modelů k systémům, které dokážou aktivně jednat v reálném světě. Ukážeme si, jak LLM mohou volat API, používat kalkulačky, spouštět kód nebo prohledávat web, což je aktuálně nejsilnější trend v automatizaci.

# 9. Bezpečnost, evaluace a limity modelů
Praktický průvodce temnější stránkou jazykových modelů a jejich limity. Probereme reálné bezpečnostní hrozby jako prompt injection, úniky citlivých dat, způsoby automatizovaného testování výstupů a obranu proti halucinacím v produkci.

# 10. Byznys nasazení a budoucí trendy
Závěrečné ukotvení v tvrdé realitě trhu. Rozebereme, kdy má smysl nasadit open-source model a kdy raději zaplatit za komerční API, jak spočítat návratnost investice a jaké teoretické koncepty se s největší pravděpodobností brzy stanou novým standardem.