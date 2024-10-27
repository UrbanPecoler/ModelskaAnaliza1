# Naloga 3: Numerična minimizacija

## Opis naloge
V tej nalogi raziskujem numerične metode za minimizacijo funkcij več spremenljivk in njihovo uporabo pri reševanju dveh fizikalnih problemov:
1. **Thomsonov problem**: Iskanje razporeditve N enakih klasičnih nabojev na površini prevodne krogle, ki minimizira elektrostatično energijo sistema. Primerjal sem učinkovitost in natančnost različnih minimizacijskih metod, kot sta Powellova metoda in n-dimenzionalni simpleks (Nelder-Mead oz. "ameba").

2. **Problem optimalne vožnje skozi semafor**: Iskanje optimalnega režima vožnje z diskretizirano časovno skalo, tako da vozilo doseže semafor ob prižigu zelene luči. Problem sem reševal z dodajanjem omejitvenih funkcij hitrosti in uporabo numerične minimizacije.

V nalogi implementiram in primerjam različne numerične metode, s poudarkom na optimizaciji hitrosti konvergence in natančnosti izračunov. Uporabljam metode kot so Nelder-Mead, Powellova metoda, metoda konjugiranih gradientov in metoda Broyden–Fletcher–Goldfarb–Shanno (BFGS). Vse so dostopne v Python knjižnici `scipy.optimize`.


## Metode numerične minimizacije
V nalogi sem preizkusil in primerjal naslednje minimizacijske metode:
- **Nelder-Mead (ameba)**: Robustna, a počasna metoda, ki temelji na krčenju simpleksa. Primerna je tudi za negladke funkcije.

- **Powellova metoda**: Primerna za gladke funkcije, uporablja konjugirane smeri za natančno in hitro optimizacijo.
- **BFGS**: Kvazi-Newtonske metoda, najbolje uporabljena, kadar so odvodi funkcije na voljo. Metoda ima hitro konvergenco in natančne rezultate.
- **Metoda konjugiranih gradientov (CG)**: Učinkovita metoda za velike probleme, ki ne zahteva shranjevanja matrike. Idealna je za probleme, kjer so znani odvodi funkcije.
- **Sequential Least Squares Programming (SLSQP)**: Močna metoda za omejitvene optimizacijske probleme, ki vključuje omejitve in minimalizacijske pogoje. Posebej uporabna za reševanje problemov s kompleksnimi omejitvami.

