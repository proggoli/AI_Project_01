# AI Einzelprojekt

Dieses Projekt untersucht die **Auswirkungen der Verspätungen der S9 (Lenzburg-Luzern)** auf den allgemeinen Zugsverkehr im Raum Luzern.
Dabei wurde die Analyse in verschiednene Schritte unterteilt.

## Ausgangslage
Die Einfahrt in Luzern ist doppelspurig. Alle Züge, die nach Luzern fahren, kommen am Betriebspunkt GTS vorbei.
In einem ersten Schritt werden die Verspätungen der S9 bei der Ankunft in Luzern genommen und untersucht, wie sich die Pünktlichkeit auf die Züge, welche eine Fahrplanmässige Abfhart in Luzern innerhalb von 10min nach der Ankunft der S9 in Luzern haben, auswirkt. Die Abfahrten dieser Züge wird als Grade 1 beschrieben.

Diese "Grade 1" Züge fahren bei ihrer Abfahrt aus dem Bahnhof Luzern beim Betriebspunkt GTS vorbei, bei welchem ihnen wiederum Züge begegnen, welche das Ziel Luzern haben. Da in GTS Einfädelungskonflikte enstehen, besteht der Verdacht, dass sich Verspätungen der Grade 1 Züge auf diese Züge auswirkt. Deshalb wird in einem zweiten Schritt untersucht, wie die Pünktlichkeit der Grade 2 Züge im Betriebspunkt GTS ist. Dabei wird ein Zeitfenster von 8min gewählt (alle Züge welche bis zu 8min später nach einer Durchfahrt des Grade 1 Zuges in GTS Richtung Luzern fahren).

Um einen Vergleich zu haben, wird die Analyse in zwei Schritten erstellt:

## Schritt 1: Analyse mithilfe GLM
es werden zwei GLM Modelle erstellt. Einmal werden die Grade 1 Züge betrachtet, in Abhängigkeit der Pünktlichkeit der S9 in Luzern und den Wetterdaten (Temperatur, Niederschlag, Wind und Schneehöhe)
Im zweiten Modell werden die Grade 2 Züge betrachtet, in Abhängigkeit der Pünktlichkeit der S9 in Luzern, der Pünktlichkeit der Grade 1 Zügen und den Wetterdaten (Temperatur, Niederschlag, Wind und Schneehöhe)

## Schritt 2: Analyse mithilfe tabPFN
in einem zweiten Schritt wird die genau gleiche Analyse erstellt, jedoch mithilfe des Modells tabPFN.

## Schritt 3: Vorhersagen
Mithilfe des tabPFN werden Vorhersagen simuliert, wie sich die Pünktlichkeit bei verschiedenen Verspätungen der S9 auf Grade 1 und Grade 2 Zügen verhält.
