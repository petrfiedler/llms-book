# 1. Od dopředných sítí k sekvencím

## 1.1 Kde končí školní teorie: Proč dopředné sítě na text prostě nestačí
Intuitivní srovnání toho, co znáte ze školy (kde má vstup vždy přesně daný počet pixelů nebo hodnot), s reálným světem textu, kde má každá věta jinou délku. Ukážeme si, proč klasická "feed-forward" síť logicky zkolabuje, když se snaží zpracovat dynamickou sekvenci, a proč jsme museli změnit celou architekturu uvažování.

## 1.2 Překlad z lidštiny do matematiky: Tokeny a Embeddingy
Jak udělat ze slova (nebo jeho části) číslo, kterému síť rozumí. Představíme si embeddingy velmi vizuálně jako souřadnice na obrovské "mapě významů" (kde jsou král a královna blízko sebe). Jasně rozlišíme, co je **zastaralý koncept** (statický Word2Vec) a jak se to dělá dnes v **SOTA (State-of-the-Art) businessu** pomocí sub-word tokenizace (BPE) a kontextových embeddingů.

## 1.3 První pokus o čtení popořadě: Recurrent Neural Networks (RNN)
Ukážeme si základní princip sítě, která si předává "paměť" (hidden state) z jednoho slova na druhé – jako když čtete knihu. Tuto architekturu si probereme jen jako **důležitý historický koncept**, u kterého si rovnou řekneme, proč by vás dnes z každé AI firmy vyhodili, kdybyste na něm chtěli stavět jazykový model (problém mizejícího gradientu a katastrofální zapomínání u delších textů).

## 1.4 Záplata jménem LSTM (a GRU): Jak jsme chvíli opravovali neopravitelné
Vysvětlení Long Short-Term Memory na jednoduchém příkladu z běžného života (představte si vyhazovače v klubu, který pouští dovnitř jen důležité informace a ty zbytečné zapomíná). Upozorníme, že ačkoliv jde o **zastaralý koncept pro generování textu**, v businessu se s ním stále můžete občas setkat u specifických úloh (např. detekce anomálií v časových řadách z průmyslových senzorů). 

## 1.5 Úzké hrdlo sekvencí: Proč to všechno muselo jít do koše
Závěr kapitoly, který tvoří můstek k dnešním LLM. Vysvětlíme si největší slabinu celého přístupu RNN/LSTM: nutnost počítat slova *jedno po druhém*. Ukážeme si, proč je tento sekvenční přístup noční můrou pro moderní grafické karty (GPU) za miliony dolarů, které potřebují paralelizovat, a proč průmysl nutně potřeboval přijít s úplně novým řešením (Transformers), které dnes vládne světu.

# 2. Transformer: Skutečný motor moderní AI

## 2.1 Pád starých králů: Proč RNN a LSTM přestaly stačit
Krátký pohled do historie na sítě, které zpracovávaly text slovo od slova. Vysvětlíme si zásadní problém sekvenčního zpracování dat, které bránilo efektivnímu tréninku na moderních GPU a představovalo úzké hrdlo (dnes již zastaralý, ale pro kontext nesmírně důležitý koncept).

## 2.2 Zrození Transformeru: Vše najednou namísto hezky popořadě
Úvod do revoluce z roku 2017 (papír "Attention Is All You Need"). Zjistíte, jak Transformer vyřešil problém úzkého hrdla tím, že "vidí" celou větu naráz, a proč byla tato změna paradigmatu klíčová pro dnešní AI průmysl a masivní škálování.

## 2.3 Self-Attention: Jak slova chápou svůj kontext
To absolutně nejdůležitější jádro Transformeru vysvětlené zcela bez matiky. Intuitivní vysvětlení na příkladech z reálného života – představíme si koncepty Query, Key a Value (QKV) jako systém hledání v chytré firemní kartotéce, díky kterému slovo "banka" pozná, jestli jde o instituci, nebo o lavičku v parku.

## 2.4 Multi-Head Attention: Víc hlav víc ví
Proč modelu nestačí dávat pozor jen jedním způsobem. Ukážeme si, jak různé "hlavy" pozornosti paralelně sledují různé vztahy v textu – zatímco jedna hlava řeší gramatickou strukturu (podmět a přísudek), druhá zachytává emoce a třetí hledá faktické souvislosti.

## 2.5 Positional Encoding: Jak neztratit orientaci v čase a prostoru
Když model čte všechna slova najednou paralelně, jak ví, jaké slovo je první a jaké poslední? Vysvětlíme si tento elegantní trik, který slovům přidává "časová razítka" a dává modelu nezbytný smysl pro pořadí.

## 2.6 Zbytek skládačky: Feed-Forward sítě a Residual Connections
Tady se napojíme na to, co už znáte ze školy. Podíváme se, co se děje se slovem poté, co díky pozornosti pochopí svůj kontext. Ukážeme si, jakou roli zde hrají klasické dopředné sítě, a proč jsou nezbytné "zkratky" (Residual Connections) a normalizace pro to, aby se síť vůbec dala natrénovat.

## 2.7 Encoder vs. Decoder: Co se reálně používá v dnešním State-of-the-Art
Původní Transformer měl dvě části (Encoder a Decoder). Jasně si vymezíme, co se dnes reálně používá v businessu – proč modely jako ChatGPT nebo Claude (SOTA) zahodily Encoder a používají pouze Decoder-only architekturu, a na co se naopak původní architektura hodí dnes.

## 2.8 Hardwarová synergie: Proč Transformer ovládl svět
Praktický pohled pod pokličku. Zjistíte, proč je architektura Transformeru tak dokonalá pro dnešní hardware. Pochopíte vztah mezi paralelizací a moderními grafickými kartami (GPU) a proč právě tento technický detail umožnil trénink dnešních gigantických LLM.

# 3. Pre-training: Zrození gigantických modelů

## 3.1 Data jako palivo: Co reálně čte dnešní AI
Zatímco ve škole jste trénovali modely na čistých a připravených datasetech (jako je MNIST), gigantické LLM se učí na surovém internetu. Tato podkapitola jednoduše vysvětlí proces sběru (např. Common Crawl), masivní deduplikace a nutné filtrace kvality. Dozvíte se, proč je dnes čistota dat důležitější než jejich objem, co jsou syntetická data (současný state-of-the-art trend) a jak firmy řeší nedostatek kvalitních textů.

## 3.2 Cíl hry zvaný Next-Token Prediction: Jak z hádání vzniká porozumění
Vysvětlíme si absolutní jádro toho, co model při pre-trainingu matematicky dělá – jednoduše se snaží uhodnout následující slovo (token). Ukážeme si intuitivně, proč tato na první pohled primitivní hra nutí síť vytvořit si ve svých vahách komplexní model světa, pochopit gramatiku, logiku i faktické znalosti, aniž by k tomu byla explicitně programována.

## 3.3 Hardware a infrastruktura: Pohled do serveroven AI gigantů
Opuštění teoretické bubliny: ukážeme si, jak vypadá fyzická realita AI byznysu. Probereme obrovské GPU clustery (tisíce karet jako NVIDIA H100 či B200) a vysvětlíme si, proč v praxi bývá úzkým hrdlem spíše komunikace mezi kartami a servery (networking, InfiniBand) než samotný výpočetní výkon.

## 3.4 Rozděl a panuj: Distribuovaný trénink a paralelizace v praxi
Nejmodernější modely se sice nevejdou do paměti jedné grafické karty, přesto je dokážeme trénovat. Představíme si průmyslový standard – 3D paralelizaci (kombinaci Data, Tensor a Pipeline parallelismu) a zmíníme techniky jako ZeRO (Zero Redundancy Optimizer). Vše vysvětleno velmi srozumitelně na analogii tovární linky, abyste přesně chápali, jak si tisíce GPU dělí práci.

## 3.5 Scaling Laws: "Fyzikální" zákony umělé inteligence
Jak firmy vědí, že model za desítky milionů dolarů bude fungovat, ještě než ho vůbec začnou trénovat? Představíme si fenomén Scaling Laws (zákony škálování) a moderní "Chinchilla" zákony. Naučíte se počítat s magickým trojúhelníkem: velikost modelu (počet parametrů) vs. objem dat vs. výpočetní výkon (Compute) a zjistíte, proč jsou starší modely podtrénované.

## 3.6 Business realita: Má pro vás smysl trénovat vlastní model od nuly?
Praktické shrnutí na závěr. Jasně oddělíme to, co dělají tech-giganti (OpenAI, Google, Meta), od toho, co dělají běžné firmy a startupy. Probereme koncept "Continual Pre-trainingu" (dotrénování na interních firemních datech) a řekneme si, kdy dává ekonomický smysl budovat doménově specifický model (jako BloombergGPT) a kdy je to jen vyhazování peněz oknem.

# 4. Alignment: Jak zkrotit umělou inteligenci

## 4.1 Proč Base Model nestačí: Monstrum pod pokličkou
Krátký úvod do toho, proč model natrénovaný jen na doplňování textu z internetu (next-token prediction) neumí přirozeně odpovídat na otázky a funguje spíše jako autokorekce na steroidech. Vysvětlení populárního konceptu "Shoggotha" a jasný rozdíl mezi Base modelem a Instruct/Chat modelem, se kterým se běžně setkáváme.

## 4.2 Supervised Fine-Tuning (SFT): Učíme model formátu a poslušnosti
Jak funguje SFT v praxi. Ukážeme si, jak modelu na stovkách či tisících prémiových ukázkách ukázat, jak přesně má vypadat ideální konverzace. Rozebereme moderní "less is more" (kvalita nad kvantitou) přístup a proč je SFT absolutní nutností pro jakékoliv firemní nasazení.

## 4.3 RLHF (Reinforcement Learning from Human Feedback): Zastaralý, ale průlomový koncept
Intuitivní vysvětlení toho, co původně odstartovalo úspěch ChatGPT. Naučíme se, jak funguje Reward Model (hodnotící model) a algoritmus PPO (Proximal Policy Optimization). Jasně si řekneme, proč se o tomto přístupu sice všude učí, ale v dnešní firemní praxi se od něj upouští, protože je příliš složitý, nestabilní a drahý.

## 4.4 DPO (Direct Preference Optimization): Současný State-of-the-Art
Vysvětlení moderní, matematicky elegantní metody, která dnes vládne open-source scéně i firemnímu prostředí. Ukážeme si, jak DPO obchází nutnost složitého Reward Modelu a učí LLM rovnou z ukázek typu "odpověď A je lepší než odpověď B". Pokud dnes firma dělá alignment, dělá ho pomocí DPO (nebo jeho variant jako KTO či ORPO).

## 4.5 Constitutional AI a RLAIF: Když umělá inteligence trénuje sama sebe
Pohled na to, jak to dělá Anthropic (tvůrci modelů Claude). Jak vyřešit problém, že lidští hodnotitelé jsou drazí a pomalí? Vytvoříme "ústavu" (sadu pravidel) a necháme jiný, chytřejší model, aby automaticky generoval preferenční data pro náš trénink. Jde o obrovský současný trend v automatizaci alignmentu.

## 4.6 Red Teaming, Jailbreaking a Guardrails: Bezpečnost v produkci
Praktický pohled na to, jak se modely nasazují v businessu. Co to znamená, když se hackeři snaží model "rozbít" pomocí Prompt Injection, jak se testuje bezpečnost (Red Teaming) a jaké ochranné bariéry (Guardrails) se před model vkládají, aby z firemního chatbota nikdy nevypadly toxické nesmysly nebo citlivá firemní data.

# 5. Anatomie SOTA: MoE a multimodalita

## 5.1 Záchrana výpočetního výkonu: Co je to Mixture of Experts (MoE)
Úvod do "řídkých" (sparse) architektur. Intuitivní vysvětlení, jak je možné, že model s obrovským počtem parametrů běží tak rychle. Rozdíl mezi klasickými "Dense" modely (kde se při každém slovu zapotí celá síť) a MoE, kde pracuje jen vybraná úzká specializace. Zjistíte, proč je toto dnes absolutní SOTA a nutný standard pro běh velkých modelů v businessu.

## 5.2 Router: Inteligentní výhybkář a anatomie expertů
Detailní pohled pod kapotu MoE. Jak síť pozná, kterého "experta" má na dané slovo zavolat? Vysvětlíme si mechanismy jako Top-K routing, nutnost balancování zátěže (load balancing) a to, co se stane, když jsou všichni tokeny posílány za jedním expertem (token dropping). Ukážeme si, co z toho je praxe a co spíše teoretický problém.

## 5.3 Evoluce smyslů: Od "slepených" modelů k nativní multimodalitě
Rozdíl mezi zastaralým (ale v menších firmách stále používaným) přístupem propojování různých modelů (např. Whisper pro řeč + LLM pro text) a SOTA nativní multimodalitou typu Any-to-Any. Jak se změní architektura, když model už od základu přijímá jakýkoliv typ vstupu.

## 5.4 Zrak LLMs: Vision Transformers (ViT) a tokenizace pixelů
Jak přesně dostaneme obrázek do transformeru, který byl původně navržený jen na text. Vysvětlení krájení obrazu na "patche", jejich převod na vektory (embeddingy) a smíchání s textovými tokeny. Ukážeme si, proč modely jako Gemini tak skvěle chápou grafy a schémata.

## 5.5 Sluch a vnímání času: Zpracování audia a videa v reálném čase
SOTA v oblasti zvuku a videa (modely jako GPT-4o nebo Gemini 1.5). Vysvětlení, jak model poslouchá nativní audio vlny (a díky tomu vnímá emoce, pauzy nebo tón hlasu místo pouhého textového přepisu). Jak se řeší video pomocí sekvence obrázků, časových embeddingů a obrovského kontextového okna.

## 5.6 MoE a Multimodalita v praxi: Co dává smysl pro business
Shrnutí toho, jak tyto architektury reálně nasadit. Kdy má smysl pro firmu rozjet vlastní open-weight MoE model (např. Mixtral), jak obrovské jsou hardwarové nároky na VRAM u MoE oproti Dense modelům a proč se na ty nejtěžší multimodální úlohy dnes téměř vždy volí komerční API (OpenAI, Google, Anthropic). Nástin budoucího vývoje a potenciálu on-device MoE modelů v telefonech.

# 6. Magie pod kapotou: Optimalizace inference

Tohle se ve škole neučí, ale pro praxi je to naprosto kritické. Vysvětlíme si, jak funguje KV cache pro plynulé generování textu a jak se dají obří modely zmenšit pomocí kvantizace tak, aby jejich provoz ve firmě nestál jmění.

## 6.1 Anatomie generování textu: Prefill vs. Decode fáze a proč je to tak pomalé
Základní stavební kámen pro pochopení inference. Intuitivní rozpad toho, jak model nejprve hromadně „přečte“ náš prompt (Prefill) a pak po jednom slovíčku potí za běhu (Decode). Vysvětlíme si zásadní produkční fakt: proč LLMs většinou nebrzdí výpočetní výkon procesoru, ale rychlost, jakou dokážeme tahat data z paměti (tzv. *memory-bound* problém).

## 6.2 KV Cache: Paměťový trik, který zachránil generování textu
Detailní, ale velmi intuitivní pohled na Key-Value Cache. Místo abychom při generování každého nového slova museli počítat celou větu znovu, model si mezivýpočty ukládá. Ukážeme si, proč je tento absolutní business standard nezbytný, ale zároveň jak rychle dokáže sežrat veškerou paměť na grafické kartě (GPU). 

## 6.3 Kvantizace (Quantization): Jak nacpat slona do ledničky
Zmenšování modelů v praxi. Opustíme akademický standard přesných desetinných čísel (FP32/FP16) a podíváme se na celočíselnou magii (INT8, INT4). Představíme si aktuální *State-of-the-Art* metody, které firmy denně používají k seškrtání nákladů na servery o desítky procent (AWQ, GPTQ, GGUF), a ukážeme si, jak je možné zmenšit model na třetinu bez toho, aniž by výrazně zhloupl.

## 6.4 FlashAttention: Když hardware běží na plné obrátky
Moderní *State-of-the-Art* nutnost. Klasická vrstva pozornosti (Attention) z původních výzkumů je pro dlouhé texty zoufale pomalá. Vysvětlíme si, jak FlashAttention chytře obchází úzká hrdla paměti přímo na křemíku grafické karty a proč se dnes v průmyslu už v podstatě žádný jiný přístup nepoužívá.

## 6.5 Continuous Batching a PagedAttention: Realita firemních API
Jak vlastně funguje ChatGPT nebo Claude pod zátěží milionů uživatelů? Ve škole se učí posílat data v pevných dávkách (static batching), což je dnes už zastaralé. Intuitivně si vysvětlíme *Continuous batching* (přidávání uživatelů za běhu) a *PagedAttention* (např. systém vLLM), což jsou dnes ty nejdůležitější mechanismy pro to, aby se firmám provoz LLM vůbec zaplatil.

## 6.6 Spekulativní dekódování (Speculative Decoding): Rychlejší text zdarma
Fascinující a vysoce aktuální SOTA technika, která obrovsky nabírá na síle. Ukážeme si chytrý trik, kdy malý, hloupý, ale bleskově rychlý model „vykřikuje“ tipy na to, jaká slova budou následovat, a obrovský, chytrý model tyto tipy jen hromadně schvaluje. Koncept, který ukazuje budoucnost optimalizace bez jakékoliv ztráty kvality výstupu.

# 7. RAG: Propojení LLM s databázemi

## 7.1 Mýtus dotrénování a zrození RAGu
Intuitivní vysvětlení, proč je "nalejvání" firemních dat do vah modelu pomocí fine-tuningu na faktické znalosti drahý a neefektivní omyl. Představení RAGu (Retrieval-Augmented Generation) jako konceptu "studenta s otevřenou učebnicí". Zhodnocení, proč se z tohoto přístupu stal absolutní byznysový standard.

## 7.2 Od textu k číslům: Embeddings a Vektorové databáze
Jak modely chápou text matematicky přes sémantické vektory a jak funguje vyhledávání podle významu. Přehled toho, co se aktuálně používá v produkci (např. Pinecone, Qdrant, pgvector), jaký je rozdíl oproti klasickým relačním databázím a proč jsou tradiční full-textová hledání (vyhledávání přesných slov) často na ústupu, ačkoli mají stále své specifické místo.

## 7.3 Naivní RAG: Základní pipeline krok za krokem
Rozborka toho nejjednoduššího RAGu, se kterým začíná každý inženýr. Projdeme si celý proces od nahrání dat (Ingestion), přes jejich rozsekání na menší kusy (Chunking), až po vyhledání relevantních pasáží (Retrieval) a finální složení promptu pro LLM (Generation). 

## 7.4 Pokročilý chunking a boj se špinavými daty (Data Ingestion)
To, co odděluje teoretické hračky od reálného byznysu, je zpracování špatně formátovaných PDFek, tabulek a firemní wiki. Ukážeme si techniky dělení textu (od pevných délek s překryvem až po moderní sémantický chunking) a vysvětlíme si, proč je příprava dat tou nejdůležitější (a nejbolestivější) částí celého procesu.

## 7.5 Advanced RAG: Jak to dělají profíci (State-of-the-Art)
Běžný naivní RAG v praxi často selhává. Zde se zaměříme na moderní state-of-the-art techniky: přepisování nepřesných uživatelských dotazů (Query Transformation), hybridní vyhledávání (kombinace vektorů a klíčových slov) a především Re-ranking (nasazení menšího modelu, který přerovná výsledky hledání podle skutečné relevance předtím, než je dostane velké LLM).

## 7.6 Evaluace: Jak dokázat, že RAG nehalucinuje
Jedna z nejtěžších byznysových výzev – jak exaktně změřit kvalitu systému, který generuje volný text. Představení moderních frameworků (jako je RAGAS) a konceptu "LLM-as-a-judge", kde k hodnocení přesnosti, relevance a opodstatněnosti (groundedness) používáme jiný AI model.

## 7.7 Agentic RAG: Budoucnost podnikového vyhledávání
Přechod od hloupé jednosměrné trubky k systémům, kde se model chová jako agent. Ukážeme si koncepty s obrovským potenciálem do budoucna, které se už nyní derou do produkce: modely, které si samy volí, do které ze tří různých databází nahlédnou, nebo si uvědomí, že jim dodané informace nestačí, a spustí doplňující vyhledávání (Multi-hop retrieval).

# 8. Agenti: Když modely používají nástroje

## 8.1 Od pasivního textového generátoru k aktivnímu agentovi
Tato podkapitola vás intuitivně provede tím, co to vlastně AI agent je. Ukážeme si rozdíl mezi modelem, který jen "předpovídá další slovo" (jako student u ústní zkoušky), a systémem, který má "ruce a oči" – dokáže si uvědomit, že něco neví, a aktivně si pro tu informaci sáhnout. Vysvětlíme si základní smyčku agenta: Myšlenka -> Akce -> Pozorování (Thought-Action-Observation).

## 8.2 Function Calling: Jak model reálně sahá do reálného světa (State-of-the-art v byznysu)
Tady se podíváme pod pokličku toho, jak přesně propojení s nástroji funguje na úrovni kódu. Žádná magie – model prostě místo běžného textu vygeneruje striktně formátovaný JSON s parametry pro nějakou vaši funkci (např. `vyhledej_pocasi(mesto="Praha")`). Toto je dnes naprostý standard (SOTA) a hlavní způsob, jakým firmy integrují LLM do svých interních systémů, CRM a databází.

## 8.3 Anatomie rozhodování: ReAct a další myšlenkové rámce (Důležitý koncept)
ReAct (Reasoning + Acting) je koncept, který odstartoval agentní revoluci. Ačkoliv dnes už modely zvládají volání funkcí nativně a čistý textový ReAct je mírně zastaralý, je to naprosto klíčový koncept pro pochopení toho, jak donutit model "přemýšlet nahlas" předtím, než provede nějakou akci. Ukážeme si, proč tohle "zastavení se a zamyšlení" radikálně snižuje chybovost.

## 8.4 Code Interpreter: LLM jako datový analytik a programátor (Nejsilnější současný trend)
Když LLM neumí spočítat složitou rovnici nebo analyzovat obří Excel, necháme ho, ať si na to napíše Python skript a sám si ho spustí v izolovaném prostředí (sandboxu). Detailně si rozebereme, jak tento mechanismus (známý z ChatGPT nebo Claude) funguje v praxi, a proč je to momentálně to nejspolehlivější, co můžete v analytickém byznysu nasadit.

## 8.5 Agenti s přístupem na web: Automatický průzkumník
Jak funguje Perplexity nebo webový agent? Není to jen RAG (vyhledávání ve vlastních datech), ale schopnost modelu samostatně formulovat dotazy pro Google/Bing, pročítat výsledky, klikat na odkazy a syntetizovat odpověď. Ukážeme si, jaké to má technické překážky a proč je web scraping pro agenty tak složitý.

## 8.6 Multi-agentní systémy: Když si AI povídá s AI (Potenciál do budoucna)
Co se stane, když postavíte tým specialistů – jedno LLM bude mít roli programátora, druhé roli testera a třetí roli manažera? Představíme si frameworky jako CrewAI nebo AutoGen. Zdůrazníme si ale, že ačkoliv je to obrovský hype a hudba budoucnosti, v tvrdé byznysové praxi se to dnes zatím často hroutí do nekonečných smyček a dává se přednost spíše striktním, předem daným procesům (tzv. state machines).

## 8.7 Bezpečnost, spolehlivost a "Human-in-the-loop" (Praktická realita)
Když dáte modelu přístup k nástrojům, může vám smazat produkční databázi nebo odeslat nesmyslný e-mail tisícům klientů. V této podkapitole probereme reálnou podnikovou praxi: proč se plně autonomní agenti v byznysu (zatím) nepoužívají, co je to "Human-in-the-loop" (člověk jako schvalovatel akcí) a jak ošetřit "halucinace", když model zavolá neexistující funkci.

# 9. Bezpečnost, evaluace a limity modelů

## 9.1 Útoky na LLMs: Prompt Injection a Jailbreaking
Základní přehled největších produkčních hrozeb. Intuitivně si ukážeme, jak funguje přímý a nepřímý prompt injection (např. škodlivý pokyn skrytý na webové stránce, kterou model sumarizuje). Rozebereme, co je aktuální state-of-the-art obrana v businessu (např. LLM Guardrails, oddělování kontextu) a proč jsou klasické metody ze security světa (jako jednoduché filtrování zakázaných slov) proti jazykovým modelům beznadějně zastaralé.

## 9.2 Ochrana citlivých dat a úniky informací (Data Leakage)
Jak zajistit, že model nezačne uživatelům recitovat tajná data vaší firmy nebo osobní údaje (PII) klientů. Zaměříme se na podnikovou praxi (enterprise SOTA) – od automatické sanitizace dat před odesláním do API až po nasazování menších open-source modelů lokálně (on-premise). Zmíníme i teoretické útoky jako *training data extraction*, kterých se v běžném RAG businessu zatím tolik bát nemusíte, ale mají ohromný význam, pokud plánujete model sami finetunovat. 

## 9.3 Zkrocení halucinací v produkčním prostředí
Pochopení toho, že halucinace jsou fundamentální vlastností predikce dalšího tokenu, nikoliv "bugem", který by šel smazat. Naučíme se, jak s nimi bojovat pomocí moderních businessových technik, jako je "Grounding" (ukotvení na realitu) v rámci RAG architektur. Vysvětlíme si, proč snaha model "přeučovat" pomocí fine-tuningu kvůli každé faktické chybě je slepá a zastaralá ulička, kterou se v praxi nemá smysl vydávat.

## 9.4 Evaluace a automatizované testování (LLM-as-a-Judge)
Jak proboha otestovat systém, jehož výstupem je volný text a neexistuje jen jedna správná odpověď? V klasickém NLP jsme používali metriky jako ROUGE nebo BLEU – ty jsou dnes pro LLMs absolutně zastaralé a v produkci nepoužitelné. Představíme si state-of-the-art přístup: "LLM-as-a-Judge", kdy využíváme jeden silný model (např. GPT-4) k bodování a automatizovanému testování výstupů vašeho menšího produkčního modelu.

## 9.5 LLM Guardrails: Nasazení bezpečnostních mantinelů
Detailní pohled na to, jak se dnes staví ochranné vrstvy mezi modelem a uživatelem (např. open-source framework NeMo Guardrails). Rozebereme mechanismy, jak v reálném čase blokovat toxický obsah, halucinace nebo snahy o konkurenční špionáž dříve, než se odpověď vůbec dostane k uživateli. Toto je dnešní absolutní standard pro jakékoliv bezpečné B2B nebo B2C nasazení.

## 9.6 Skutečné hard limity dnešních architektur a pohled do budoucna
Intuitivní a upřímné shrnutí toho, co dnešní Transformer modely prostě matematicky nezvládnou (např. spolehlivé logické plánování na mnoho kroků dopředu). Rozebereme si fenomén "Lost in the Middle" (model zapomene, co bylo uprostřed obřího kontextu) a ukážeme si, co jsou teoretické koncepty (jako Test-Time Compute modely či alternativní architektury jako Mamba), které mají do budoucna potenciál tyto limity prorazit, ale zatím nejsou na úrovni běžného firemního nasazení.

# 10. Byznys nasazení a budoucí trendy

Závěrečné ukotvení v tvrdé realitě trhu. Rozebereme, kdy má smysl nasadit open-source model a kdy raději zaplatit za komerční API, jak spočítat návratnost investice a jaké teoretické koncepty se s největší pravděpodobností brzy stanou novým standardem.

## 10.1 Open-source vs. Komerční API (Kdy stavět a kdy si připlatit)
Intuitivní průvodce klíčovým rozhodnutím každého projektu. Kdy dává smysl posílat data přes API modelům jako jsou GPT-4 či Claude (nejrychlejší cesta na trh, ale hrozí vendor lock-in a únik dat), a kdy je naopak nutností sáhnout po open-source modelech typu Llama či Mistral a hostovat si je na vlastních serverech (maximální soukromí a kontrola, ale vysoké nároky na infrastrukturu).

## 10.2 Ekonomika LLM: Jak spočítat reálné náklady a ROI
Matematika za byznysem. Jak správně počítat cenu za vstupní a výstupní tokeny, jak odhadnout náklady na GPU servery při vlastním hostování a proč RAG architektura často prodražuje dotazy. Přidáme jednoduchý framework pro výpočet návratnosti investice (ROI), abyste uměli projekt obhájit před managementem.

## 10.3 LLMOps a produkční architektura: SOTA v reálném byznysu
Nasazením to teprve začíná. Jak firmy reálně řídí desítky tisíc dotazů denně. Ukážeme si moderní "LLM routing" (dynamické směrování jednoduchých dotazů na levné modely a složitých na ty drahé), sémantické cachování (aby se model neptal dvakrát na to samé) a jak se v produkci monitorují a omezují halucinace či prompt injekce.

## 10.4 Co je mrtvé, co je standard a co je teprve hudba budoucnosti
Jasné shrnutí technologií pro rychlou orientaci. Co už nedělat (např. trénovat model od nuly bez miliardového rozpočtu), co je naprostý denní chléb firem (RAG, prompt inženýrství, API), co je drahé ale funkční (fine-tuning specifických úloh) a co jsou zatím jen akademické koncepty, které v produkci spíše selhávají, ale mají obrovský potenciál.

## 10.5 Posun k menším modelům (SLMs) a multi-agentním systémům
Pohled za roh dnešních trendů. Proč se byznys začíná odklánět od gigantických modelů k vysoce optimalizovaným "Small Language Models" (SLMs), které běží levně a lokálně. Jak fungují sítě autonomních agentů, kteří si práci předávají mezi sebou, a jaké další objevy v blízké budoucnosti redefinují způsob, jakým dnes AI aplikace stavíme.

# 11. Bonusový projekt I: Jak postavit vlastní Perplexity (Real-time Web Search Agent)

V této kapitole propojíme znalosti o RAGu, agentech a optimalizaci inference. Ukážeme si, jak postavit systém, který dokáže v reálném čase analyzovat dotaz, prohledat internet, vyextrahovat relevantní informace a vygenerovat citovanou odpověď. Nejde o hračku, ale o produkční SOTA architekturu s ohledem na latenci a streaming.

## 11.1 Architektura Search Agenta a Query Rewriting
Jak funguje orchestrace celého procesu. Proč se uživatelský dotaz neposílá rovnou do vyhledávače, ale nejdříve ho rychlý model (např. Llama 3 8B nebo Claude 3 Haiku) přeformuluje do několika paralelních search queries. Ukázka promptu pro query expansion.

## 11.2 Real-time vyhledávání a scraping v praxi
Integrace s APIs jako Tavily, Bing Search nebo Serper. Jak efektivně a asynchronně scrapovat text z webových stránek, jak řešit anti-bot ochrany a jak z webového HTML rychle vysekat čistý text pro kontext modelu.

## 11.3 Reranking a in-memory RAG
Internet vrací obrovské množství textu, které se nevejde do kontextového okna (nebo by generování bylo moc drahé a pomalé). Jak aplikovat cross-encoder reranker (např. Cohere Rerank) na scrapované weby a vybrat jen ty nejrelevantnější odstavce v řádu milisekund.

## 11.4 Syntéza odpovědi, citace a streaming UI
Jak donutit model (např. GPT-4o nebo Claude 3.5 Sonnet), aby přesně citoval své zdroje. Implementace Server-Sent Events (SSE) pro streamování tokenů uživateli v reálném čase, aby aplikace působila bleskurychle (perceived latency), a jak do streamu vkládat odkazy. 

## 11.5 Produkční nasazení, sémantická cache a monitoring
Nasazení aplikace (FastAPI + React/Next.js). Využití sémantické cache (např. Redis + embeddings) pro ušetření API kreditů u často kladených dotazů. Sledování halucinací a latence pomocí nástrojů jako Langfuse nebo LangSmith.


# 12. Bonusový projekt II: Jak postavit Copilota ve VS Code (Coding Assistant s kontextem)

Praktický návod na stavbu vlastního programátorského asistenta podobného GitHub Copilotu nebo Cursoru. Naučíš se pracovat s modely trénovanými speciálně na kód, řešit Fill-in-the-Middle (FIM) doplňování a budovat kontext z obří lokální codebase.

## 12.1 Výběr modelu a Fill-in-the-Middle (FIM) generování
Jaký je rozdíl mezi chatovacím modelem a code-completion modelem (StarCoder, CodeLlama, DeepSeek Coder). Jak fungují FIM tokeny (<PRE>, <MID>, <SUF>) a jak správně poskládat prompt, aby model doplnil kód přesně tam, kde bliká kurzor.

## 12.2 Budování kontextu: AST, LSP a vektorizace codebase
Model nemůže znát celý tvůj projekt. Jak parsovat Abstract Syntax Tree (AST), jak využít Language Server Protocol (LSP) pro získání definic funkcí a jak vektorizovat lokální repozitář (lokální ChromaDB/Qdrant), aby model věděl o funkcích z jiných souborů.

## 12.3 Prompt Engineering pro kód a Tree-of-Thought
Jak strukturovat systémový prompt pro vývojářského agenta. Ukázky SOTA technik pro komplexní refactoring, kdy model nejprve napíše plán změn (Chain-of-Thought) a až poté generuje samotný kód.

## 12.4 Architektura VS Code extenze a Guardrails
Jak propojit TypeScriptovou extenzi s Python backendem. Kontrola vygenerovaného kódu proti syntax errorům předtím, než se ukáže uživateli. Ochrana proti vložení škodlivého kódu (security guardrails).

## 12.5 Hostování vlastního code modelu a Continuous Batching
Pokud nechceš platit za API, jak nasadit vlastní code model na vlastní GPU serveru. Praktická ukázka využití vLLM, Continuous Batching a FlashAttention pro dosažení stovek vygenerovaných tokenů za sekundu.


# 13. Bonusový projekt III: Lokální AI pro pražský LegalTech startup (Privacy-First)

Představ si, že pracuješ ve startupu, který analyzuje české právní smlouvy. Data jsou extrémně citlivá, nesmí opustit firmu a API OpenAI nepřipadá v úvahu. Jak postavit kompletní lokální SOTA řešení, které mluví česky a běží na firemním hardwaru.

## 13.1 Výběr open-weights modelu a české specifikum
Srovnání dostupných modelů (Llama 3 8B, Mistral Nemo, Qwen) pro češtinu. Proč už dnes nepotřebujeme nutně modely trénované výhradně v ČR, ale stačí kvalitní multilingvální base model a dobrý prompt.

## 13.2 Kvantizace pro lokální běh (GGUF, AWQ, EXL2)
Jak dostat 70B model na jeden dostupný server bez toho, abychom zruinovali startup. Detailní návod na kvantizaci modelů do 4-bit nebo 8-bit přes GGUF (pro CPU/Apple Silicon s Ollamou) a AWQ/EXL2 (pro rychlou inferenci na dedikovaných Nvidiích).

## 13.3 Lokální RAG a Privacy-first Embeddingy
Nasazení lokálního embedding modelu (např. E5-multilingual nebo BGE) pro vektorizaci dokumentů, které nikdy neopustí intranet. Architektura offline RAGu nad PostgreSQL s pgvector nebo lokální instancí Qdrantu.

## 13.4 Finetuning pro doménová specifika (LoRA / QLoRA)
Když promptování nestačí: Jak připravit dataset ze starých smluv či diagnóz a jak levně a efektivně doladit model pomocí Parameter-Efficient Fine-Tuning (PEFT) s využitím QLoRA, aniž bychom potřebovali H100 cluster.

## 13.5 Ekonomika hardwaru a LLMOps pro On-Premise
Koupit vlastní GPU rig do kanceláře, nebo pronajmout bare-metal server v EU? Rozbor nákladů (TCO). Jak řešit verzování vlastních LoRA adaptérů, audity, logování dotazů a zajištění 100% data compliance dle GDPR/AI Actu.