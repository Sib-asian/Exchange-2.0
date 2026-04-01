#!/usr/bin/env python3
"""Genera documento PDF con quadro generale automazione Exchange-2.0"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# Registra font
pdfmetrics.registerFont(TTFont('SimHei', '/usr/share/fonts/truetype/chinese/SimHei.ttf'))
pdfmetrics.registerFont(TTFont('Microsoft YaHei', '/usr/share/fonts/truetype/chinese/msyh.ttf'))
pdfmetrics.registerFont(TTFont('Times New Roman', '/usr/share/fonts/truetype/english/Times-New-Roman.ttf'))
registerFontFamily('SimHei', normal='SimHei', bold='SimHei')
registerFontFamily('Microsoft YaHei', normal='Microsoft YaHei', bold='Microsoft YaHei')
registerFontFamily('Times New Roman', normal='Times New Roman', bold='Times New Roman')

# Crea documento
doc = SimpleDocTemplate(
    "/home/z/my-project/download/Quadro_Automazione_Exchange_2.0.pdf",
    pagesize=A4,
    title="Quadro Automazione Exchange-2.0",
    author="Z.ai",
    creator="Z.ai",
    subject="Analisi opportunita di automazione software Exchange-2.0"
)

# Stili
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'TitleStyle',
    fontName='Microsoft YaHei',
    fontSize=24,
    leading=32,
    alignment=TA_CENTER,
    spaceAfter=24
)

h1_style = ParagraphStyle(
    'H1Style',
    fontName='Microsoft YaHei',
    fontSize=16,
    leading=22,
    alignment=TA_LEFT,
    spaceBefore=18,
    spaceAfter=12,
    textColor=colors.HexColor('#1F4E79')
)

h2_style = ParagraphStyle(
    'H2Style',
    fontName='Microsoft YaHei',
    fontSize=13,
    leading=18,
    alignment=TA_LEFT,
    spaceBefore=12,
    spaceAfter=8,
    textColor=colors.HexColor('#2E75B6')
)

body_style = ParagraphStyle(
    'BodyStyle',
    fontName='SimHei',
    fontSize=10.5,
    leading=16,
    alignment=TA_LEFT,
    wordWrap='CJK',
    spaceAfter=8
)

cell_style = ParagraphStyle(
    'CellStyle',
    fontName='SimHei',
    fontSize=9,
    leading=12,
    alignment=TA_LEFT,
    wordWrap='CJK'
)

header_style = ParagraphStyle(
    'HeaderStyle',
    fontName='Microsoft YaHei',
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    textColor=colors.white
)

story = []

# Copertina
story.append(Spacer(1, 100))
story.append(Paragraph("Quadro Generale Automazione", title_style))
story.append(Spacer(1, 12))
story.append(Paragraph("Exchange-2.0", title_style))
story.append(Spacer(1, 48))
story.append(Paragraph("Analisi completa delle opportunita di automazione", body_style))
story.append(Paragraph("e miglioramento del software di analisi scommesse", body_style))
story.append(Spacer(1, 100))
story.append(Paragraph("Data: 31 Marzo 2026", body_style))
story.append(PageBreak())

# 1. Panoramica del Progetto
story.append(Paragraph("1. Panoramica del Progetto", h1_style))
story.append(Paragraph(
    "Exchange-2.0 e un software avanzato per l'analisi probabilistica di partite di calcio, "
    "progettato per supportare decisioni di betting informate. Il sistema utilizza un approccio "
    "multi-modello che combina tre metodologie statistiche (Poisson Bivariata, Copula Frank, "
    "Catena di Markov) per calcolare probabilita su vari mercati: 1X2, Over/Under, BTTS, "
    "Asian Handicap e Correct Score.",
    body_style
))
story.append(Paragraph(
    "L'architettura del software e divisa in due pagine principali: la pagina Principale, "
    "dedicata all'analisi approfondita di singole partite con input manuali delle linee di "
    "mercato, e la pagina Scanner, progettata per l'analisi automatica multi-partita da URL "
    "Nowgoal. Il sistema estrae automaticamente dati da fonti esterne tramite OCR da screenshot "
    "e parsing di pagine web, implementando un flusso semi-automatico che riduce significativamente "
    "il lavoro manale dell'utente.",
    body_style
))

# 2. Stato Attuale dell'Automazione
story.append(Paragraph("2. Stato Attuale dell'Automazione", h1_style))
story.append(Paragraph("2.1 Funzionalita Automatizzate", h2_style))
story.append(Paragraph(
    "Il software dispone gia di un solido livello di automazione che copre diversi aspetti "
    "fondamentali del processo di analisi. Di seguito vengono dettagliate le aree gia completamente "
    "automatizzate con relativo livello di maturita e affidabilita.",
    body_style
))

# Tabella funzionalita automatizzate
auto_data = [
    [Paragraph('<b>Funzionalita</b>', header_style), Paragraph('<b>Descrizione</b>', header_style), Paragraph('<b>Stato</b>', header_style)],
    [Paragraph('Estrazione URL Nowgoal', cell_style), Paragraph('H2H, Standings, Last6, Quote 1X2 via Regex (gratuito)', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('OCR Screenshot Quote', cell_style), Paragraph('Estrazione quote da screenshot (z-ai, Gemini, OpenAI)', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('OCR Statistiche Live', cell_style), Paragraph('Tiri, corner, possesso, cartellini da screenshot', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('Scanner Multi-Partita', cell_style), Paragraph('Analisi parallela di N URL con ThreadPoolExecutor', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('Calcolo Probabilita', cell_style), Paragraph('3 modelli statistici (Poisson, Copula, Markov)', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('Calcolo xG Bayesiano', cell_style), Paragraph('Blend linee mercato + tiri live + forma', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('Generazione Segnali', cell_style), Paragraph('Segnali betting con edge e stake Kelly', cell_style), Paragraph('Completo', cell_style)],
    [Paragraph('Salvataggio Partite', cell_style), Paragraph('Session storage con restore stato completo', cell_style), Paragraph('Completo', cell_style)],
]

auto_table = Table(auto_data, colWidths=[120, 280, 60])
auto_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(Spacer(1, 12))
story.append(auto_table)
story.append(Spacer(1, 6))
story.append(Paragraph("Tabella 1: Funzionalita attualmente automatizzate", ParagraphStyle('Caption', fontName='SimHei', fontSize=9, alignment=TA_CENTER)))

# 2.2 Input Manuali Richiesti
story.append(Paragraph("2.2 Input Manuali Richiesti", h2_style))
story.append(Paragraph(
    "Nonostante il buon livello di automazione, alcune informazioni devono ancora essere "
    "inserite manualmente dall'utente. Questi input rappresentano opportunita di miglioramento "
    "per rendere il flusso completamente automatico.",
    body_style
))

manual_data = [
    [Paragraph('<b>Input Manuale</b>', header_style), Paragraph('<b>Dove</b>', header_style), Paragraph('<b>Priorita Automazione</b>', header_style)],
    [Paragraph('AH Apertura/Corrente', cell_style), Paragraph('Pagina Principale', cell_style), Paragraph('Alta - disponibile in Vs_hOdds', cell_style)],
    [Paragraph('Total Apertura/Corrente', cell_style), Paragraph('Pagina Principale', cell_style), Paragraph('Alta - disponibile in Vs_hOdds', cell_style)],
    [Paragraph('Minuto partita (live)', cell_style), Paragraph('Pagina Principale', cell_style), Paragraph('Media - richiede integrazione API', cell_style)],
    [Paragraph('Punteggio live', cell_style), Paragraph('Pagina Principale', cell_style), Paragraph('Media - richiede integrazione API', cell_style)],
    [Paragraph('Quote Exchange', cell_style), Paragraph('Sezione Avanzata', cell_style), Paragraph('Alta - API Betfair disponibile', cell_style)],
]

manual_table = Table(manual_data, colWidths=[140, 100, 220])
manual_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(Spacer(1, 12))
story.append(manual_table)
story.append(Spacer(1, 6))
story.append(Paragraph("Tabella 2: Input che richiedono intervento manuale", ParagraphStyle('Caption', fontName='SimHei', fontSize=9, alignment=TA_CENTER)))

# 3. Opportunita di Automazione
story.append(Paragraph("3. Opportunita di Automazione", h1_style))

# 3.1 Estrazione Dati Mancanti
story.append(Paragraph("3.1 Estrazione Dati Mancanti da Nowgoal", h2_style))
story.append(Paragraph(
    "La pagina Nowgoal contiene diversi dati utili che non vengono attualmente estratti. "
    "Questi dati potrebbero migliorare significativamente la precisione del modello. "
    "L'analisi della struttura della pagina ha identificato tre categorie di dati mancanti "
    "con alto potenziale predittivo.",
    body_style
))

story.append(Paragraph("<b>A. h_data / a_data (Partite Recenti)</b>", body_style))
story.append(Paragraph(
    "Nowgoal espone le statistiche delle ultime partite giocate da ciascuna squadra, "
    "incluse le sequenze di gol fatti e subiti. Questo dato permetterebbe di calcolare "
    "un xG derivato dalle performance reali recenti, piu affidabile delle medie stagionali. "
    "Il modello potrebbe identificare trend di forma (squadra in crescendo o calando) e "
    "integrarli come fattore correttivo. Attualmente la dataclass PrematchAnalysisExtracted "
    "ha gia i campi home_recent_results, away_recent_results, home_form_trend e away_form_trend "
    "ma non sono popolati dall'estrazione.",
    body_style
))

story.append(Paragraph("<b>B. Vs_hOdds (Quote Multi-Bookmaker)</b>", body_style))
story.append(Paragraph(
    "La sezione quote di Nowgoal mostra le linee AH e Total da diversi bookmaker con "
    "apertura e chiusura. Questo permetterebbe di estrarre automaticamente le linee "
    "di apertura/corrente senza input manuale. Inoltre, il movimento delle linee "
    "(line movement) e un segnale prezioso: movimento significativo indica spesso "
    "'sharp money' o informazioni privilegiate. I campi sono gia definiti nella dataclass "
    "(ah_line_open, total_line_open, line_movement_ah) ma non estratti.",
    body_style
))

story.append(Paragraph("<b>C. Punti Classifica (Motivazione)</b>", body_style))
story.append(Paragraph(
    "I punti in classifica permettono di calcolare il fattore motivazione. Una squadra "
    "in lotta salvezza o per il titolo ha motivazione alta, mentre una squadra gia salva "
    "a meta classifica ha motivazione bassa. Questo fattore puo influenzare le probabilita "
    "fino al 5-10%. I campi home_points, away_points, home_motivation, away_motivation "
    "sono gia presenti nella dataclass ma non valorizzati.",
    body_style
))

# Tabella priorita implementazione
prio_data = [
    [Paragraph('<b>Dato</b>', header_style), Paragraph('<b>Impatto Previsto</b>', header_style), Paragraph('<b>Complessita</b>', header_style), Paragraph('<b>Raccomandazione</b>', header_style)],
    [Paragraph('Vs_hOdds (AH/Total)', cell_style), Paragraph('Alto - elimina input manuali', cell_style), Paragraph('Media', cell_style), Paragraph('Implementare subito', cell_style)],
    [Paragraph('h_data/a_data', cell_style), Paragraph('Alto - xG da partite recenti', cell_style), Paragraph('Media', cell_style), Paragraph('Implementare subito', cell_style)],
    [Paragraph('Punti Classifica', cell_style), Paragraph('Medio - fattore motivazione', cell_style), Paragraph('Bassa', cell_style), Paragraph('Implementare a breve', cell_style)],
    [Paragraph('Team Names', cell_style), Paragraph('Basso - solo display', cell_style), Paragraph('Bassa', cell_style), Paragraph('Gia implementato', cell_style)],
]

prio_table = Table(prio_data, colWidths=[100, 150, 80, 130])
prio_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(Spacer(1, 12))
story.append(prio_table)
story.append(Spacer(1, 6))
story.append(Paragraph("Tabella 3: Priorita di implementazione dati mancanti", ParagraphStyle('Caption', fontName='SimHei', fontSize=9, alignment=TA_CENTER)))

# 3.2 Integrazione API Esterne
story.append(Paragraph("3.2 Integrazione API Esterne", h2_style))
story.append(Paragraph(
    "L'integrazione con API di terze parti puo trasformare il software da semi-automatico "
    "a completamente automatico, eliminando la necessita di inserire URL o fare screenshot. "
    "Questa sezione analizza le principali opzioni disponibili con relativi costi e benefici.",
    body_style
))

story.append(Paragraph("<b>A. Betfair Exchange API</b>", body_style))
story.append(Paragraph(
    "L'API di Betfair Exchange permette di ottenere quote in tempo reale per tutti i mercati. "
    "Questo eliminerebbe completamente la necessita di input manuali per le quote. L'API "
    "fornisce anche il volume di liquidita su ogni quota, utile per valutare l'affidabilita "
    "del mercato. Costo: gratuito per uso personale (fino a 1000 req/hour), richiede account "
    "Betfair attivo. Implementazione stimata: 2-3 giorni per integrazione base.",
    body_style
))

story.append(Paragraph("<b>B. API-Football / Football-Data.org</b>", body_style))
story.append(Paragraph(
    "Queste API forniscono statistiche dettagliate sulle partite: formazioni, statistiche "
    "live, storico H2H, classifica aggiornata. Possono integrarsi con lo Scanner per "
    "automatizzare completamente l'acquisizione dati pre-match. Costo: tier gratuito "
    "(100 req/giorno), tier pro a partire da 10 euro/mese. Implementazione stimata: "
    "3-5 giorni per integrazione completa.",
    body_style
))

story.append(Paragraph("<b>C. Pinnacle API</b>", body_style))
story.append(Paragraph(
    "Pinnacle offre un feed di quote altamente efficiente con linee AH e Total molto "
    "precise. L'API e riservata a partner ma esistono wrapper non ufficiali. Le quote "
    "Pinnacle sono considerate il benchmark del mercato per la loro efficienza. "
    "Implementazione: complessa per restrizioni API, alternativa e usare API di comparazione.",
    body_style
))

# 3.3 Automazione Monitoraggio
story.append(Paragraph("3.3 Automazione Monitoraggio e Alert", h2_style))
story.append(Paragraph(
    "Attualmente l'utente deve manualmente avviare le analisi e controllare i risultati. "
    "Un sistema di monitoraggio automatico potrebbe scansionare periodicamente le partite "
    "e inviare notifiche quando vengono rilevati segnali interessanti.",
    body_style
))

story.append(Paragraph("<b>Funzionalita Suggerite:</b>", body_style))
story.append(Paragraph(
    "Sistema di scheduler per scansioni automatiche periodiche, ad esempio ogni 30 minuti "
    "per le partite della giornata. Alert via email o Telegram quando vengono rilevati "
    "segnali con alta confidenza (es. 3 stelle). Dashboard con riepilogo giornaliero "
    "delle opportunita identificate. Tracking automatico dei risultati per validare "
    "le previsioni e calibrare il modello. Storico delle analisi con esiti reali per "
    "migliorare continuamente la precisione.",
    body_style
))

# 3.4 Machine Learning Enhancements
story.append(Paragraph("3.4 Machine Learning Enhancements", h2_style))
story.append(Paragraph(
    "Il modello attuale si basa su metodologie statistiche classiche. L'integrazione "
    "di tecniche di machine learning potrebbe migliorare la calibrazione e l'accuratezza "
    "delle previsioni.",
    body_style
))

story.append(Paragraph("<b>Opzioni di Miglioramento:</b>", body_style))
story.append(Paragraph(
    "Calibrazione Isotonica: gia implementata, puo essere estesa con modelli piu sofisticati. "
    "Feature Engineering: utilizzare i dati estratti (forma recente, motivazione, movement) "
    "come feature per un modello supervisato. Ensemble Learning: combinare i 3 modelli "
    "attuali con pesi adattivi basati su performance storiche per mercato. Time-Series "
    "Model: modellare l'evoluzione delle probabilita durante la partita per identificare "
    "punti di ingresso ottimali.",
    body_style
))

# 4. Roadmap Consigliata
story.append(Paragraph("4. Roadmap Consigliata", h1_style))
story.append(Paragraph(
    "Sulla base dell'analisi effettuata, si propone una roadmap di implementazione "
    "prioritizzata che massimizzi l'impatto con sforzo contenuto.",
    body_style
))

roadmap_data = [
    [Paragraph('<b>Fase</b>', header_style), Paragraph('<b>Attivita</b>', header_style), Paragraph('<b>Tempo</b>', header_style), Paragraph('<b>Impatto</b>', header_style)],
    [Paragraph('1. Breve Termine', cell_style), Paragraph('Estrarre Vs_hOdds (AH/Total) da Nowgoal', cell_style), Paragraph('2-3 giorni', cell_style), Paragraph('Alto - elimina input manuale', cell_style)],
    [Paragraph('2. Breve Termine', cell_style), Paragraph('Estrarre h_data/a_data per forma recente', cell_style), Paragraph('2-3 giorni', cell_style), Paragraph('Alto - migliora xG', cell_style)],
    [Paragraph('3. Medio Termine', cell_style), Paragraph('Integrazione Betfair API per quote live', cell_style), Paragraph('3-5 giorni', cell_style), Paragraph('Molto Alto - automazione totale', cell_style)],
    [Paragraph('4. Medio Termine', cell_style), Paragraph('Sistema alert automatici (Telegram)', cell_style), Paragraph('2-3 giorni', cell_style), Paragraph('Medio - user experience', cell_style)],
    [Paragraph('5. Lungo Termine', cell_style), Paragraph('API-Football per dati completi', cell_style), Paragraph('5-7 giorni', cell_style), Paragraph('Alto - fonte alternativa', cell_style)],
    [Paragraph('6. Lungo Termine', cell_style), Paragraph('Tracking risultati e validazione', cell_style), Paragraph('3-5 giorni', cell_style), Paragraph('Alto - miglioramento continuo', cell_style)],
]

roadmap_table = Table(roadmap_data, colWidths=[80, 200, 60, 120])
roadmap_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#E6F3FF')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.white),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#E6F3FF')),
    ('BACKGROUND', (0, 4), (-1, 4), colors.white),
    ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#E6F3FF')),
    ('BACKGROUND', (0, 6), (-1, 6), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(Spacer(1, 12))
story.append(roadmap_table)
story.append(Spacer(1, 6))
story.append(Paragraph("Tabella 4: Roadmap di implementazione consigliata", ParagraphStyle('Caption', fontName='SimHei', fontSize=9, alignment=TA_CENTER)))

# 5. Conclusioni
story.append(Paragraph("5. Conclusioni", h1_style))
story.append(Paragraph(
    "Il progetto Exchange-2.0 dispone gia di una solida base di automazione che copre "
    "gli aspetti piu complessi dell'analisi probabilistica. Tuttavia, esistono significative "
    "opportunita di miglioramento che possono trasformare il software da strumento "
    "semi-automatico a sistema completamente autonomo.",
    body_style
))
story.append(Paragraph(
    "Le priorita immediate dovrebbero concentrarsi sull'estrazione dei dati mancanti "
    "da Nowgoal (Vs_hOdds e h_data/a_data), che richiedono sforzo limitato ma hanno "
    "alto impatto operativo. Successivamente, l'integrazione con API esterne (Betfair, "
    "API-Football) eliminerebbe la necessita di input manuali, rendendo il flusso "
    "completamente automatico.",
    body_style
))
story.append(Paragraph(
    "Infine, l'implementazione di un sistema di monitoraggio e alert trasformerebbe "
    "il software da strumento di analisi a vero e proprio assistente automatico, "
    "capace di notificare opportunita senza intervento umano. Questo rappresenterebbe "
    "il passo finale verso un sistema di betting algoritmico completo.",
    body_style
))

# Build PDF
doc.build(story)
print("PDF generato con successo!")
