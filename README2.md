# En resa genom maskinlärningens historia

AI-utvecklingen målas ofta upp som en rad stora, magiska språng. Ena året har vi enkla beslutsträd, nästa år resonerar modeller flytande på doktorandnivå. Men bakom rubrikerna döljer sig en mer jordnära ingenjörsfråga: **vad tillför egentligen varje nytt paradigm i mätbar träffsäkerhet – och vad kostar det?**

Inspirerad av ett projekt av [Ed Donner](https://github.com/ed-donner/llm_engineering/tree/main/week6) ville jag göra en tidsresa genom maskininlärningens utveckling och se vad varje steg faktiskt tillför. Därför ställde jag modeller från olika generationer mot exakt samma problem, med samma data och samma måttstock – från enkel linjär regression och klassiska beslutsträd till specialtränade djupa neurala nät och toppmoderna språkmodeller.

Själva uppgiften var avsiktligt renodlad: **kan en modell förutsäga konsumentpriset på en Amazon-produkt utifrån enbart dess beskrivning?**

Priserna i sig är inte poängen. De fungerar som en gemensam måttstock – ett sätt att göra decenniers AI-utveckling direkt jämförbar i ett och samma experiment.

* **Måttstock:** Genomsnittligt absolut prisfel (MAE, *Mean Absolute Error*) i dollar. Ju lägre siffra, desto vassare modell.
* **Databas:** 820 000 förädlade Amazon-produkter.
* **Notering:** Resultaten gäller förutsättningarna i detta experiment. Den mänskliga referensen omfattar 100 produkter, och Deep NN-diagrammet visar en separat Lite-körning (huvudresultatet avser full körning).

---

## Innehåll

- [Rond 1 – En mänsklig referens och de blinda modellerna](#rond-1)
- [Grundarbetet – Skräp in ger skräp ut](#grundarbetet)
- [Rond 2 – När modellen får orden (Bag-of-Words)](#rond-2)
- [Rond 3 – Klassisk ML: Trädens revansch](#rond-3)
- [Rond 4 – Frontier-modeller mot ett skräddarsytt neuralt nätverk](#rond-4)
- [Rond 5 – Den dyrköpta läxan om fine-tuning](#rond-5)
- [Samlad resultattavla](#resultat)
- [Beslutsmatris och strategiska lärdomar](#lardomar)
- [Vad resan säger om AI:s utveckling](#slutsats)
- [Kod och originalnotebooks](#notebooks)

---

<a id="rond-1"></a>
## Rond 1: En mänsklig referens och de blinda modellerna

Innan några algoritmer släpptes lösa behövdes en mänsklig ribba. Hur bra är en erfaren person på att uppskatta vad produkter kostar bara genom att läsa beskrivningen?

Ed Donner satte sig ner och läste beskrivningarna för 100 slumpmässigt utvalda Amazon-artiklar och noterade sina gissningar. Facit visade att han i genomsnitt felade med **87,62 dollar per produkt**.

- **Resultat (Människa):** I genomsnitt **$87,62** i fel.

![Mänskliga prisgissningar](assets/human.png)

*Varje punkt representerar en produkt. Den vågräta axeln visar verkligt pris och den lodräta visar gissningen. Den streckade diagonalen är en perfekt träff – ju längre bort från linjen en punkt hamnar, desto större är felet.*

Detta är vår mänskliga ribba. Människan förstår kontext, varumärkesprestige och skillnaden mellan kolfiber och plast – men är sämre på att hålla tusentals disparata nischpriser i huvudet. Hur långt ifrån den ribban startar matematiken?

### Första jämförelsen: Gissa alltid på medelpriset

Den enklast tänkbara modellen är helt blind. Den struntar fullständigt i produktbeskrivningen och svarar konsekvent med katalogens medelpris: 140,57 dollar.

- **Resultat (Gissa medelpris):** Ett snittfel på **$106,18**.

Om den mänskliga gissningen ($87,62) är riktmärket är detta projektets absoluta bottennivå. Varje seriös arkitektur måste prestera betydligt bättre än så här för att ha ett existensberättigande.

### Linjär regression med enkla variabler

Går det att förbättra gissningen genom att använda en klassisk linjär regressionsmodell och mata den med två till synes mätbara egenskaper: **produktens vikt** och **antalet tecken i beskrivningen**?

- **Resultat (Enkel regression):** Ett snittfel på **$101,56**.

![Linjär regression med stela variabler](assets/linear_regression.png)

*Med endast vikt och beskrivningens längd som indata kan regressionen i princip bara lägga en helt platt linje strax under medelvärdet.*

Resultatet gav knappt fem dollars förbättring. Men det är inte regressionsmodellens fel – det är vi som gav den usla förutsättningar. Vikten på ett paket avslöjar sällan om det innehåller billig kattsand eller en dyr kamera. Och som diagrammet nedan visar finns det absolut ingen korrelation mellan hur mycket en säljare skriver och vad varan kostar:

![Pris mot textlängd](assets/pris_mot_textlangd.png)

*Det finns inget linjärt mönster mellan textmängd och pris. Vissa av de dyraste produkterna har korta specifikationer, medan billiga prylar ofta har långa säljtexter.*

Experimentets första fundamentala insikt var ett faktum: **Det är sällan modellen som är flaskhalsen – det är informationen du ger den.** Utan tillgång till själva orden i texten förblir algoritmerna hjälplösa.

<details><summary><strong>Teknisk fördjupning: Så sattes den enkla regressionsmodellen upp</strong></summary>

Först extraherades de två numeriska egenskaperna: produktens vikt och textlängden på dess sammanfattning. En binär flagga skapades för att explicit hantera saknade viktvärden:

```python
def get_features(item):
 return {
 "weight": item.weight,
 "weight_unknown": 1 if item.weight == 0 else 0,
 "text_length": len(item.summary)
 }
```

Genom att sätta `weight_unknown = 1` tillåts modellen lära sig en separat koefficient för produkter där vikten saknas, istället för att felaktigt tolka 0 som fjäderlätt.

Datan samlades i en DataFrame och tränades med standardiserad linjär regression i scikit-learn:

```python
from sklearn.linear_model import LinearRegression

feature_columns = ['weight', 'weight_unknown', 'text_length']
X_train = train_df[feature_columns]
y_train = train_df['price']

model = LinearRegression()
model.fit(X_train, y_train)
```

Modellen optimerar vikterna för dessa tre tal via minsta kvadratmetoden. Men eftersom varken vikt eller teckenantal bär någon stabil prissignal kan modellen inte göra annat än att justera baslinjen marginellt kring medelvärdet.

Fullständig kod: [Steg 3 – klassisk maskininlärning](notebooks/steg3.html).

</details>

---

<a id="grundarbetet"></a>
## Grundarbetet: Skräp in ger skräp ut

För att modellerna ska kunna förstå vad produkterna faktiskt är måste vi ge dem tillgång till texten. Men här ställdes vi inför maskininlärningens äldsta naturlag: **skräp in ger skräp ut** – och rå e-handelsdata är extremt stökig.

Vi utgick från ett dataset från Hugging Face med beskrivningar av cirka 3 miljoner produkter. Vid en manuell granskning hittades till exempel en mikrovågsugn för **21 000 dollar**. Det visade sig vara en professionell TurboChef-ugn för restaurangkök. Priset var korrekt, men en sådan extrem outlier skulle förvränga modellernas inlärningskurvor. Prisspannet avgränsades därför till mer normala konsumentnivåer: **0,50 till 999,49 dollar**.

### Ett balanserat och viktat urval

Att bara rensa bort fel och dubbletter räckte inte. Om billiga produkter och ett fåtal jättelika kategorier dominerar materialet får modellerna aldrig se tillräckligt med exempel på mer sällsynta eller dyrare varor.

Därför gjordes ett **viktat slumpmässigt urval** på **820 000 produkter**. Dyrare produkter gavs högre sannolikhet att väljas, samtidigt som de dominerande kategorierna *Tools and Home Improvement* och *Automotive* skalades ner.

<p align="center">
 <img src="assets/categories.png" width="36%" alt="Produktkategorier">
 <img src="assets/prisfordelning.png" width="62%" alt="Prisfördelning">
</p>

*Kategoriernas fördelning efter det viktade urvalet (vänster) samt den resulterande prisfördelningen (höger). Genom att vikta urvalet fick modellerna ett brett, differentierat underlag över hela prisskalan.*

<details><summary><strong>Teknisk fördjupning: Matematiken bakom det viktade urvalet</strong></summary>

Priserna normaliserades först till ett intervall mellan 0 och 1. Därefter applicerades en kvadratisk funktion för att ge högre priser kraftigt ökad sannolikhet, samtidigt som kategorifilter dämpade överrepresenterade segment:

```python
import numpy as np

np.random.seed(42)
SIZE = 820_000

prices = np.array([it.price for it in items], dtype=float)
categories = np.array([it.category for it in items])

# Normalisera priserna mellan 0 och 1 (1e-9 skyddar mot nolldivision)
p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)

# Kvadratisk viktning prioriterar dyrare produkter
w = p**2
w[categories == "Tools_and_Home_Improvement"] *= 0.5
w[categories == "Automotive"] *= 0.05

# Normalisera vikterna så att summan blir 1.0 (sannolikhetsfördelning)
w = w / w.sum()

# Dra ett slumpmässigt urval utan återläggning
idx = np.random.choice(len(items), size=SIZE, replace=False, p=w)
sample = [items[i] for i in idx]
```

Genom att använda `p**2` ser vi till att en produkt för 500 dollar har avsevärt större chans att väljas än en för 5 dollar, vilket motverkar den naturliga anhopningen av billiga småartiklar.

Fullständig kod: [Steg 1 – dataurval och tvätt](notebooks/steg1.html).

</details>

### Kontextdesign: LLM:en som digital redaktör

Produktbeskrivningar på Amazon är notoriskt spretiga: HTML-taggar, säljsnack, versaler och interna artikelnummer blandas i en enda röra.

Innan vi tränade en enda prismodell lät vi därför en liten, snabb språkmodell (**GPT-4.1 Nano**) agera digital redaktör. Genom en strikt instruktion tvingades den strukturera varje oordnad råtext till fem rena, standardiserade fält: **Title, Category, Brand, Description och Details**.

Detta kallas *kontextdesign*. För en bråkdel av ett öre per produkt ($0,00006) pressades hundratusentals spretiga poster in i en enhetlig form. Genom att lyfta bort bruset i förväg skapade vi optimala förutsättningar för alla kommande modeller.

<details><summary><strong>Teknisk fördjupning: Systemprompt och tokenekonomi för GPT-4.1 Nano</strong></summary>

För att säkerställa att språkmodellen inte svävade iväg användes en komprimerad systemprompt som tvingade fram strikt radformatering:

```python
SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""
```

Ett verkligt utdrag från processningen visar den dramatiska komprimeringen:

```text
Title: Schlage F59 Interior Half Door Knob – Oil Rubbed Bronze
Category: Hardware
Brand: Schlage
Description: A solid bronze interior knob with deadbolt for easy, secure entry.
Details: Features oil-rubbed finish, 4" center-to-center clearance, and lifetime warranty.

Input tokens: 446 | Output tokens: 86 | Kostnad: 0,006 cent
```

Denna förädling halverade brusnivån och säkerställde att textvektoriseringen i nästa rond fick arbeta med ren, innehållsrik information.

Fullständig kod: [Steg 2 – LLM-förädling](notebooks/steg2.html).

</details>

---

<a id="rond-2"></a>
## Rond 2: När modellen får orden (Bag-of-Words)

Hur får man en matematisk formel att läsa text? Den klassiska metoden heter **Bag-of-Words**.

Vi byggde en lista med de 2 000 vanligaste och mest betydelsebärande orden i våra nystädade produktbeskrivningar. Varje produkt översattes därefter till en vektor – en rad med 2 000 siffror – som anger hur ofta ord som *luxury*, *LED*, *leather*, *plastic* eller *Schlage* förekommer.

Sedan körde vi exakt samma enkla linjära regression som nyss – men nu med orden som indata.

- **Resultat (Bag-of-Words):** Felet rasar till **$76,81**.

![Linjär regression med Bag-of-Words](assets/nlp_linear_regression.png)

*Med Bag-of-Words som representation börjar gissningarna äntligen leta sig upp mot den perfekta diagonalen.*

Detta är ett otroligt ögonblick i experimentet. En enkel regressionsmodell som tränas och körs på en bråkdel av en sekund har precis **slagit den mänskliga referensen ($87,62) med över 10 dollar**. Inte för att modellen blivit smartare, utan för att den äntligen fick relevanta signaler att arbeta med: enskilda ord bär på enorma prissignaler.

Men metoden har en uppenbar svaghet: den är helt **kontextblind**. För en Bag-of-Words-modell ser ordet *fil* exakt likadant ut oavsett om texten handlar om en *datorfil*, en *körfil på motorvägen* eller en skål med *frukostfil*. Orden kastas i en säck där ordningsföljd och samspel ignoreras.

Frågan hänger kvar: *Vad händer när vi låter modellerna förstå relationer och mönster mellan orden?*

<details><summary><strong>Teknisk fördjupning: Från text till 2 000 dimensioner med CountVectorizer</strong></summary>

Omvandlingen görs via `CountVectorizer` i scikit-learn med engelska stoppord bortfiltrerade:

```python
from sklearn.feature_extraction.text import CountVectorizer

np.random.seed(42)
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(documents)
```

`fit_transform` bygger ett ordförråd av de 2 000 vanligaste orden och skapar en gles matris (*sparse matrix*) där varje rad representerar en produkt och varje kolumn representerar ett ord. 

När en linjär regression tränas på denna matris tilldelas varje unikt ord en vikt i dollar. Ord som "diamond" eller "leather" får positiva koefficienter som höjer prisgissningen, medan ord som "sticker" eller "plastic" drar ner den.

Fullständig kod: [Steg 3 – klassisk maskininlärning](notebooks/steg3.html).

</details>

---

<a id="rond-3"></a>
## Rond 3: Klassisk ML – Trädens revansch

En linjär modell kan bara addera eller subtrahera priser för enskilda ord. Men i verkligheten beror ords betydelse på andra ord: "gaming" tillsammans med "laptop" signalerar en helt annan prislapp än "gaming" tillsammans med "mousepad".

För att fånga sådana samspel vände vi oss till ensemblemodeller av beslutsträd.

Först ut var en **Random Forest** (slumpmässig skog av beslutsträd), som tog oss ner till:

- **Resultat (Random Forest):** Ett snittfel på **$72,28**.

Därefter testade vi **XGBoost** (Gradient Boosting). Till skillnad från Random Forest, som bygger oberoende träd parallellt, tränar XGBoost sina träd sekventiellt – där varje nytt träd specialiserar sig på att korrigera de misstag som de tidigare träden gjorde.

- **Resultat (XGBoost):** Felet pressades ner till **$68,23**.

![Resultat för XGBoost](assets/xgboost.png)

*XGBoost lyckas fånga komplexa, icke-linjära samband i texten, även om de allra dyraste lyxprodukterna fortfarande är svåra att pricka helt rätt.*

I en tid när nästan allt handlar om generativ AI är detta värt att begrunda. En klassisk, beprövad modell har kapat det ursprungliga felet med nästan 40 dollar. Den tränas snabbt, körs lokalt på millisekunder för en bråkdel av en cent och har ett helt förutsägbart, deterministiskt beteende.

Men hur nära toppen kan man komma utan djupare språkförståelse?

<details><summary><strong>Teknisk fördjupning: Träning och inferens med XGBoost</strong></summary>

XGBoost matades med samma 2 000-dimensionella ordmatris som regressionen:

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
 n_estimators=1000,
 random_state=42,
 n_jobs=4,
 learning_rate=0.1
)
xgb_model.fit(X, prices)
```

Modellen bygger 1 000 träd med en inlärningstakt på 0,1. Vid prediktion transformeras texten genom det etablerade ordförrådet, och resultatet kapslas in med en säkerhetsspärr mot negativa priser:

```python
def xg_boost(item):
 x = vectorizer.transform([item.summary])
 return max(0, float(xgb_model.predict(x)[0]))
```

Styrkan hos XGBoost ligger i dess förmåga att skapa förgreningar på ordkombinationer (t.ex. *IF "stainless" == 1 AND "steel" == 1 AND "commercial" == 1 THEN +$150*).

Fullständig kod: [Steg 3 – klassisk maskininlärning](notebooks/steg3.html).

</details>

---

<a id="rond-4"></a>
## Rond 4: Frontier-modeller mot ett skräddarsytt neuralt nätverk

Nu kliver vi in i den moderna djupinlärningens era. Här ställde vi två helt olika filosofier mot varandra: **världens största generella språkmodeller** mot **våra egna specialtränade neurala nätverk**.

### De generella jättarna (Zero-shot)

Vi bad världens ledande språkmodeller att uppskatta priset på produkterna utifrån beskrivningen – helt utan att ha tränat en sekund på vårt dataset. De fick förlita sig enbart på den allmänna världsförståelse de byggt upp under sin förträning.

Resultaten ritade omedelbart om kartan:

- **GPT-4.1 Nano:** $62,51 i fel
- **Grok 4.1 Fast:** $57,62 i fel
- **Gemini 3 Pro:** $50,54 i fel
- **Claude 4.5 Sonnet:** $47,10 i fel
- **GPT-5.1:** **$44,74 i fel**

![Frontiermodeller utan träning](assets/frontiermodeller.png)

*De generella språkmodellerna presterar exceptionellt bra enbart på sin breda förståelse av marknaden.*

Här får vår tidigare "fil"-fråga sitt svar. Transformers och moderna språkmodeller är byggda för att låta orden väga och färga varandra. De förstår att "solid bronze" i en dörrknopp motiverar ett högt pris, medan "bronze finish" på en plastdetalj inte gör det. De förstår varumärkens marknadspositionering direkt ur lådan.

Men bäst i test har ett pris: GPT-5.1 är enorm, relativt långsam och dyr att köra i produktion.

Det leder till nästa ingenjörsfråga: **Hur nära jättarna kan vi komma med en egen, lokal modell som slipper betala för generell intelligens?**

<details><summary><strong>Teknisk fördjupning: Zero-shot prompt och API-anrop</strong></summary>

Till skillnad från de egna modellerna kräver frontier-modellerna minimal kod. Produkten kapslas in i en rak prompt:

```python
def messages_for(item):
 message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
 return [{"role": "user", "content": message}]

def gpt_4__1_nano(item):
 response = completion(model="openai/gpt-4.1-nano", messages=messages_for(item))
 return response.choices[0].message.content
```

Ingen anpassning, inga viktuppdateringar. Modellen resonerar sig fram till priset baserat på sin inbyggda kunskapsbas.

Fullständig kod: [Steg 4 – neurala nät och språkmodeller](notebooks/steg4.html).

</details>

### Utmanarna: Våra egna neurala nätverk

Som motvikt byggde vi två egna nätverk från grunden i PyTorch och tränade dem på vårt dataset.

1. **Vårt enkla nätverk (MLP):** Ett klassiskt åttalagers feed-forward-nätverk med drygt **669 000 parametrar**. Det landade på **$63,97** – vilket i princip matchade GPT-4.1 Nano, trots att det är en flugviktare i sammanhanget.
2. **Vårt djupa nätverk (Deep NN):** Utrustat med residualblock, LayerNorm och dropout, klockade det in på **289 miljoner parametrar**.

Jämfört med frontier-modellernas hundratals miljarder parametrar är 289 miljoner fortfarande en dvärg. Men den hade en avgörande fördel: **den slipper kunna något annat.** Den kan inte skriva sonetter eller generera kod – dess enda existensberättigande är att förstå prissättningen på Amazon.

- **Resultat (Eget djupt neuralt nät):** Ett snittfel på **$46,49** vid full träning på hela datasetet.

![Deep Neural Network – egen Lite-körning](assets/deep_nn_lite.png)

*Diagrammet visar en provkörning på ett mindre Lite-dataset ($72,54). Vid full träning på samtliga 820 000 produkter nådde modellen hela vägen ner till $46,49.*

Detta är ett slående resultat. Vår skräddarsydda specialistmodell körde rakt in mellan Gemini 3 Pro ($50,54) och Claude 4.5 Sonnet ($47,10), mindre än två dollar bakom flaggskeppet GPT-5.1. 

Det bevisar kraften i **domänspecifik specialisering**: en fokuserad modell kan matcha modeller som är tusen gånger större.

<details><summary><strong>Teknisk fördjupning: MLP-arkitektur och träningsloop i PyTorch</strong></summary>

För att mata nätverket med orddata utan att behöva underhålla ett gigantiskt ordförråd i minnet användes en `HashingVectorizer` med 5 000 dimensioner:

```python
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

vectorizer = HashingVectorizer(n_features=5000, stop_words='english', binary=True)
X = vectorizer.fit_transform(documents)
```

Det mindre nätverket definierades med 8 linjära lager och ReLU-aktiveringar:

```python
class NeuralNetwork(nn.Module):
 def __init__(self, input_size):
 super(NeuralNetwork, self).__init__()
 self.layer1 = nn.Linear(input_size, 128)
 self.layer2 = nn.Linear(128, 64)
 self.layer3 = nn.Linear(64, 64)
 self.layer4 = nn.Linear(64, 64)
 self.layer5 = nn.Linear(64, 64)
 self.layer6 = nn.Linear(64, 64)
 self.layer7 = nn.Linear(64, 64)
 self.layer8 = nn.Linear(64, 1)
 self.relu = nn.ReLU()

 def forward(self, x):
 x = self.relu(self.layer1(x))
 x = self.relu(self.layer2(x))
 x = self.relu(self.layer3(x))
 x = self.relu(self.layer4(x))
 x = self.relu(self.layer5(x))
 x = self.relu(self.layer6(x))
 x = self.relu(self.layer7(x))
 return self.layer8(x)
```

Träningen genomfördes med Adam optimizer och MSE-loss i en klassisk träningsloop över minibatches:

```python
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
 model.train()
 for batch_X, batch_y in train_loader:
 optimizer.zero_grad()
 outputs = model(batch_X)
 loss = loss_function(outputs, batch_y)
 loss.backward()
 optimizer.step()
```

Den separata djupa modellen (289M parametrar) utökade denna princip med bredare lager, residualkopplingar för att motverka gradientförsvinnande samt LayerNorm och Dropout för regularisering.

Fullständig kod: [Steg 4 – neurala nät och språkmodeller](notebooks/steg4.html) samt [Extra – djupt neuralt nätverk](notebooks/deep_neural_network_extra.html).

</details>

---

<a id="rond-5"></a>
## Rond 5: Den dyrköpta läxan om fine-tuning

Efter att ha sett hur bra GPT-4.1 Nano presterade i grundutförande ($62,51) och hur långt specialisering tog vårt eget nätverk ($46,49) verkade nästa steg självklart: **Vi gör en fine-tuning av GPT-4.1 Nano på vårt dataset.** Då borde vi få det ultimata verktyget – språkmodellens världsförståelse kombinerad med datasetets specifika priskalibrering.

Vi förberedde träningsdatan, startade finjusteringen via API:t och körde första utvärderingen.

Första körningen gav ett genomsnittligt fel på **exakt 0,00 dollar**.

Hade vi byggt en perfekt AI? Självklart inte. Den erfarne utvecklaren känner omedelbart igen symptomet: en **dataläcka**. Vid skapandet av testprompterna hade målvariabeln (priset) av misstag råkat ligga kvar i indatan. Modellen var inte synsk – den läste bara innantill från provfrågan.

När buggen åtgärdades och modellen utvärderades på ett isolerat testset kom det verkliga resultatet:

- **GPT-4.1 Nano (Grundutförande):** $62,51 i fel
- **GPT-4.1 Nano (Fine-tuned):** **$75,91 i fel**

![Fine-tuning-resultat](assets/fine_tuning_resultat.png)

*Den finjusterade modellen presterade i slutändan över 13 dollar sämre än vad den gjorde i sitt grundutförande.*

Hur kunde ytterligare träning göra modellen sämre?

Detta är en klassisk fälla inom maskininlärning som kallas **katastrofal glömska** (*catastrophic forgetting*). När man finjusterar en generell modell på en repetitiv uppgift med ett stelt svarsformat riskerar man att skriva över de välbalanserade inre representationer som modellen byggt upp under förträningen. Modellen blev så fokuserad på att imitera det specifika svarsformatet att den tappade bort sitt sunda förnuft kring varumärken och materialkvalitet.

Lärdomen är glasklar: **Fine-tuning är inte en automatisk uppgradering.** Det är ett kirurgiskt ingrepp som kräver minutiös kalibrering – och det bör aldrig vara ditt första drag.

<details><summary><strong>Teknisk fördjupning: JSONL-data och konfiguration av OpenAI fine-tuning</strong></summary>

Träningsdatan formaterades till JSONL där varje rad representerar en komplett dialog med prompt och förväntat svar:

```python
def messages_for(item):
 message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
 return [
 {"role": "user", "content": message},
 {"role": "assistant", "content": f"${item.price:.2f}"}
 ]
```

Finjusteringsjobbet startades med OpenAIs officiella SDK:

```python
from openai import OpenAI
client = OpenAI()

job = client.fine_tuning.jobs.create(
 training_file=train_file.id,
 validation_file=validation_file.id,
 model="gpt-4.1-nano-2025-04-14",
 seed=42,
 hyperparameters={"n_epochs": 1, "batch_size": 1},
 suffix="pricer"
)
```

Till skillnad från prompt engineering justeras modellens faktiska vikter under processen. Om träningsmängden saknar tillräcklig språklig variation drabbas modellen lätt av överanpassning till presentationsformen snarare än den underliggande logiken.

Fullständig kod: [Steg 5 – fine-tuning](notebooks/steg5.html).

</details>

---

<a id="resultat"></a>
## Samlad resultattavla

Här är hela startfältet sammanställt, sorterat från lägst till högst felmarginal:

![Slutlig jämförelse](assets/slutresultat.png)

*Genomsnittligt absolut prisfel (MAE) i dollar för samtliga metoder – lägre är bättre. Deep NN avser fullkörningen ($46,49), inte Lite-körningen ($72,54).*

| Modell / Metod | MAE ($) | Kommentar & Analys |
| :--- | :--- | :--- |
| 🥇 **GPT-5.1** | **$44,74** | **Bäst i test.** Extrem precision, men dyrast och tyngst i drift. |
| 🥈 **Djupt neuralt nätverk (Eget)** | **$46,49** | **Projektets stjärna.** 289M-specialist som nästan når toppen. |
| 🥉 Claude 4.5 Sonnet | $47,10 | Enorm träffsäkerhet direkt ur lådan utan specifik träning. |
| Gemini 3 Pro | $50,54 | Stabil och balanserad prestanda bland frontier-modellerna. |
| Grok 4.1 Fast | $57,62 | Snabb, effektiv och förvånansvärt vass bland jättarna. |
| GPT-4.1 Nano (Grundmodell) | $62,51 | Imponerande stark basnivå för en liten språkmodell. |
| Neuralt nätverk (Eget, 8 lager) | $63,97 | Extremt kompakt (669k parametrar), matchar GPT-4.1 Nano. |
| **XGBoost** | **$68,23** | **Bästa klassiska ML.** Blixtsnabb, billig och deterministisk. |
| Random Forest | $72,28 | Robust trädbaserad metod, men slås tydligt av gradient boosting. |
| GPT-4.1 Nano (Fine-tuned) | $75,91 | Varningsexemplet: överträning raderade den breda kunskapen. |
| Linjär regression (Bag-of-Words) | $76,81 | Bevisar att rätt textrepresentation slår mänsklig intuition. |
| **Människa (Ed Donner)** | **$87,62** | **Slagen av nästan alla algoritmer på systematisk data.** |
| Linjär regression (enkla variabler) | $101,56 | Skräp in, skräp ut – vikt och textlängd saknar prissignal. |
| **Blind gissning (Medelpris)** | **$106,18** | Projektets absoluta bottennivå. |

---

<a id="lardomar"></a>
## Beslutsmatris och strategiska lärdomar

Om du bygger AI-funktioner eller leder ett utvecklingsteam idag visar experimentet att valet av teknik sällan handlar om att "välja den senaste modellen". Det handlar om att matcha arkitekturen med din tillgängliga data och dina affärskrav.

```
                 ┌───────────────────────────────┐
                 │ Har du gott om egen ren data? │
                 └───────────────┬───────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼ JA                              ▼ NEJ
  ┌───────────────────────────┐     ┌───────────────────────────┐
  │ Krävs millisekundsvar och │     │ Behöver du djup förståelse│
  │    minimal driftkostnad?  │     │ direkt "out-of-the-box"?  │
  └─────────────┬─────────────┘     └─────────────┬─────────────┘
                │                                 │
       ┌────────┴────────┐               ┌────────┴────────┐
       ▼ JA              ▼ NEJ           ▼ JA              ▼ NEJ
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │   XGBoost    │ │  Eget djupt  │ │ Frontier-LLM │ │   Kontext-   │
 │ (Trädmodell) │ │  neuralt nät │ │  (Zero-shot) │ │ design + BoW │
 └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Fem ingenjörsprinciper från projektet

1. **En glasklar utvärdering (Eval) är A och O:** Utan ett hårt, mätbart facit fastnar man i subjektiva diskussioner om vad som "känns bäst". Det var vår strikta mätning som avslöjade dataläckan direkt och som bevisade att finjusteringen faktiskt förstörde värde.
2. **Klassisk maskininlärning är långt ifrån död:** Att XGBoost når ett fel på $68 för en bråkdel av en cent är ett oerhört starkt affärscase. I skarp produktion är 80 % av precisionen till 0,1 % av kostnaden ofta den bästa affären.
3. **Använd LLM:er som förädlare i pipelinen:** Att låta GPT-4.1 Nano tvätta och strukturera råtexten *innan* den nådde prismodellerna var projektets mest lönsamma drag. Investera i kontextdesign innan du rör modellarkitekturen.
4. **Specialisering slår storlek på hemmaplan:** En modell med 289 miljoner parametrar som bara kan en sak kan utmana modeller med hundratals miljarder parametrar – med full datakontroll och utan externa API-beroenden.
5. **Var skeptisk mot fine-tuning:** Finjustering är inte en universallösning. Optimera alltid dina prompter, din kontext och din datakvalitet innan du börjar justera modellvikter.

---

<a id="slutsats"></a>
## Vad resan säger om AI:s utveckling

Resan genom maskininlärningens utveckling gav en mer nyanserad bild av vad framsteg inom AI egentligen innebär. De stora sprången i precision kom sällan från att byta till en marginellt nyare modell. De kom från bättre förutsättningar: från dåliga variabler till meningsfulla ord, och från kaotisk råtext till strukturerad kontext.

Framgångsrik AI handlar i slutändan om sund ingenjörskonst: börja med att förstå problemet, ge modellen rätt signaler och låt hårda mätningar styra besluten. Då blir modellvalet inte en gissningslek, utan en ren avvägning mellan precision, latens och kostnad – där du vet exakt hur mycket komplexitet du faktiskt behöver betala för.

---

<a id="notebooks"></a>
## Kod och originalnotebooks

Samtliga körningar finns dokumenterade i projektets exporterade Jupyter Notebooks. HTML-filerna kan laddas ner och öppnas i valfri webbläsare för att granska källkod, grafer och körningsloggar:

- [Steg 1 – Dataurval och tvätt](notebooks/steg1.html)
- [Steg 2 – LLM-förädling och kontextdesign](notebooks/steg2.html)
- [Steg 3 – Klassisk maskininlärning (Regression, BoW, XGBoost)](notebooks/steg3.html)
- [Steg 4 – Neurala nät och språkmodeller](notebooks/steg4.html)
- [Steg 5 – Fine-tuning](notebooks/steg5.html)
- [Extra – Djupt neuralt nätverk (PyTorch 289M)](notebooks/deep_neural_network_extra.html)
- [Samlad resultatnotebook](notebooks/resultat.html)
