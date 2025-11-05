Nervenzellen - Neuronale Netze
----------------------------

Gewichte
--------

| Alter | Grösse | Geschlecht | L |
|-------|--------|------------|---|
| 30    | 170    | W          | 1 |
| 81    | 165    | W          | 0 |
| 18    | 190    | m          | 0 |

--> Matrixtabelle

[] (m,n)

Linsen für Fotoobjektive
------------------------
Mehrere Linsen --> mehrere Layers nacheinander
Es geht Lebendigkeit verloren bei zu vielen Linsen / Layers / komptizierten Linsen

Wie kommt in nächten Layer -> Matrixmultiplikation

[m,3] -> [3,5]
[5,6] x [6,20]

Linear(20,30) -> Matrix mit [20,30]

Datenmatrix  Gewichtsmatrix
[1,2]  x    [5 7 8]
[3,4]       [6 8 9]

= 1x5 + 2x6 = 17
  1x7 + 2x8 = 23
  1x8 + 2x9 = 26

Gewichte können Vektoren verstärken oder unterdrücken

Je breiter (mehr Spalten), desto weniger Bias aber mehr Varianz

Lineare Funktion, aber nicht alles ist rechteckig und linear, deshalb gibt es die Aktivierungsfunktion (auf jedem Layer).

Aktivierungsfunktion
--------------------

- ReLu = Rektified Linear Union
_/ (positive werte 1:1 durch, negative werte auf 0)

- LeakyReLu = erlaubt kleine negative Werte

- Tanh = Hyperbolic Tangens
Asymtotisch gegen 1 und -1

- Sigmoid = von 0 bis 1

Bias
----

Bias (b) verschiebt die Aktivierung nach links/rechts und legt fest, ab wann das Neuron aktiviert wird.
y = b + x x W

act 0 = b + x x w
out = b1 + xw2

L = MSE(daten,output)
keine Aktivierung beim Output! (sonst wird Ergebnis zerstört)


log

Daten von gestern label von morgen


