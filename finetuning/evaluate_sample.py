from bert_score import score

def compute_custom_bertscore(candidate_text, reference_text, lang_code):
    """
    Computes BERTScore using the specific configuration:
    - Model: xlm-roberta-large
    - Layer: 17 (This is the recommended layer for xlm-roberta-large,
      though the text mentions num_layers=24, usually the best layer is picked 
      automatically or set to ~17 for roberta-large).
    - Rescaling: True (uses official baselines)
    """
    
    # 1. Wrap inputs in lists
    cands = [candidate_text]
    refs = [reference_text]

    # 2. Compute the score
    # model_type: Explicitly set to xlm-roberta-large
    # num_layers: The model has 24 layers. Setting this usually tells the scorer 
    #             to use the embedding from that specific layer depth.
    # lang: Essential for fetching the correct baseline file for rescaling.
    P, R, F1 = score(
        cands, 
        refs, 
        model_type="xlm-roberta-large", 
        num_layers=24,  # Use embeddings from the 24th layer
        rescale_with_baseline=True, 
        lang=lang_code, # e.g., "en", "fr", "de"
        verbose=True
    )
    
    return P.item(), R.item(), F1.item()

# --- Example Usage ---
reference = """
Art. 185 Abs. 3 BV; Art. 5 Abs. 2 der Covid-19-Verordnung Erwerbsausfall; Entschädigung für den Erwerbsausfall einer selbständigerwerbenden Person aufgrund des Coronavirus; Gesetz- und Verfassungsmässigkeit der Bestimmungen der Covid-19-Verordnung Erwerbsausfall betreffend den Betrag und die Berechnung der Entschädigung in den verschiedenen zeitlich massgebenden Versionen.  Sinn und Zweck von Art. 5 Abs. 2 der Covid-19-Verordnung Erwerbsausfall in der Fassung vom 6. Juli 2020 und der Art. 5 Abs. 2bis und 2ter derselben Verordnung in der Fassung vom 8. Oktober 2020, wie sie ab dem 17. September 2020 in Kraft standen, ist es festzusetzen, dass das Taggeld aufgrund der Steuerdaten für das Jahr 2019 zu berechnen ist, welche der Verwaltung spätestens bis zum 16. September 2020 vorgelegen haben müssen (E. 11.3.2). Während die Fassung der Verordnung vom 6. Juli 2020 aufgrund der Dringlichkeit der damaligen Situation (E. 9) eine verfassungsrechtliche Immunität geniesst, ist die ab 17. September 2020 in Kraft gestandene Fassung von dieser nicht gedeckt und verstösst gegen den Grundsatz der Rechtsgleichheit (E. 11, insbesondere E. 11.4).
"""
candidate = """Art. 5 Abs. 2, 2bis und 2ter COVID-19-Verordnung; Art. 8 BV; Art. 185 Abs. 3 BV; Art. 15 COVID-19-Gesetz; Art. 11 Abs. 1 EOG; Art. 7 Abs. 1 EO-VO

E. 6–11

Die COVID-19-Erwerbsersatzregelung unterscheidet verfassungsrechtlich zwischen der ersten pandemiebedingten Notphase (17. März bis 16. September 2020) und der nachfolgenden Periode ab 17. September 2020. Der in Art. 5 Abs. 2 COVID-19-Verordnung (Fassung vom 6. Juli 2020) vorgesehene Stichtag vom 16. September 2020 für die Einreichung einer neueren Steuerveranlagung und eines Neuberechnungsgesuchs ist aufgrund der ausserordentlichen Dringlichkeit der damaligen Situation sachlich gerechtfertigt und mit dem Gleichbehandlungsgebot nach Art. 8 BV vereinbar (E. 11.3.3, 11.4).

Demgegenüber verstossen Art. 5 Abs. 2bis und 2ter COVID-19-Verordnung in der Fassung vom 8. Oktober 2020 gegen Art. 8 BV, soweit sie für den Zeitraum nach dem 16. September 2020 bei bereits laufendem Leistungsbezug eine Neuberechnung der Taggelder unter Ausschluss später verfügbarer Steuerdaten generell verbieten. Ab Herbst 2020 fehlte es an einer vergleichbaren Dringlichkeit, welche diese starre Ungleichbehandlung von Selbstständigerwerbenden mit früher bzw. später vorliegender Steuerveranlagung rechtfertigen könnte (E. 11.3.4, 11.4).

Soweit die COVID-19-Verordnung ab dem 17. September 2020 auf Art. 15 COVID-19-Gesetz beruht, geniesst sie keine verfassungsrechtliche Immunität, da das Gesetz keine hinreichend bestimmte inhaltliche Vorgabe zur Ausgestaltung des Erwerbsersatzes enthält; die Verordnungsbestimmungen unterliegen daher der vollen verfassungsrechtlichen Kontrolle (E. 9).

Die Regelung ist zudem systemwidrig, soweit sie vom in Art. 11 Abs. 1 EOG und Art. 7 Abs. 1 EO-VO verankerten Grundsatz abweicht, wonach bei nachträglicher Festsetzung eines anderen massgebenden AHV-pflichtigen Einkommens eine Neuberechnung der Entschädigung möglich sein muss (E. 11.3.4)."""

# Note: You must specify the language code (e.g., 'fr', 'en') 
# so the library knows which baseline file to download.
p, r, f1 = compute_custom_bertscore(candidate, reference, lang_code="fr")

print(f"Rescaled F1 Score: {f1:.4f}")