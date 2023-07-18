import math

from scipy.stats import hypergeom
import scipy.stats

def binom(n, k):
    return math.comb(n, k)

def calculate_binomial_probability():
    print("Binominal Verteilung B(k|p, n)")

    print("(k) Anzahl an Erfolgen:")
    success_amount = int(input())

    print("(n) Anzahl der Versuche eingeben:")
    sample_size = int(input())

    print("(p) Wahrscheinlichkeit für Treffer eingeben:")
    success_probability = (float(input().replace(",", ".")))

    var = sample_size * success_probability * (1 - success_probability)
    erwartungswert = sample_size * success_probability
    probability = binom(sample_size, success_amount) * (success_probability**success_amount) * ((1-success_probability)**(sample_size-success_amount))

    print("Var(x): " + str(round(var,2)) + "\nE(x): " + str(erwartungswert) + "\nP(X = " + str(success_amount) + "): " + str(round(probability,4)) + "\n")

    input("\nEingabe Taste drücken um fortzufahren...")


def calculate_poison_probability():
    print("Poison Verteilung λ:")
    print("(λ) Rate der Verteilung:")
    rate = float(input().replace(",", "."))

    print("(k) Stichprobengröße:")
    sample_size = int(input())

    probability = ((rate**sample_size)/math.factorial(sample_size)) * math.exp(-rate)

    print("\nErwartungswert: " + str(rate))
    print("Var(x): " + str(rate))
    print("Wahrscheinlichkeit: " + str(round(probability, 4)))

    input("\nEingabe Taste drücken um fortzufahren...")

def calculate_hypergeometric_probability():
    print("Hypergeometrische Verteilung: h(k|N,M,n)")
    # Input parameters
    print("(N) Populations Größe:")
    population_size = int(input())

    print("(M) Anzahl der Positiven Ergebnisse:")
    success_states_population = int(input())

    print("(n) Anzahl der Versuche:")
    sample_size = int(input())

    print("(k) Anzahl der Treffer:")
    observed_success_states_sample = int(input())

    # Calculate the probability
    probability = hypergeom.pmf(observed_success_states_sample, population_size, success_states_population, sample_size)
    excpected = sample_size * (population_size/success_states_population)
    var = excpected * (1 - population_size/success_states_population) * ((population_size-sample_size)/(population_size - 1))

    # Return the probability
    print("Var(x): " + str(round(var, 2)) + "\nE(x): " + str(round(excpected, 2)) + "\nP(X = " + str(observed_success_states_sample) + "): " + str(round(probability, 4)) + "\n")

    input("\nEingabe Taste drücken um fortzufahren...")


def z_zest():
    print("\nZ-Test")

    print("(x̄) Mittelwert der Stichprobe:")
    sample = float(input().replace(",", "."))

    print("(μ) Mittelwert der Population:")
    population = float(input().replace(",", "."))

    print("(σ) Standard-Abweichung:")
    std = float(input().replace(",", "."))

    print("(n) Umfang der Stichprobe (Falls nicht vorhanden \"0\" eingeben!):")
    sample_size = int(input())

    result = 0.0
    alpha_ninefive = 1.96
    alpha_ninenine = 2.5758

    if sample_size == 0:
        result = (sample - population)/std
        kiLower = population - alpha_ninefive * std
        kiUpper = population + alpha_ninefive * std
        kiLowerTwo = population - alpha_ninenine * std
        kiUpperTwo = population + alpha_ninenine * std
    else:
        result = (sample - population)/(std/math.sqrt(sample_size))
        kiLower = population - alpha_ninefive * (std/math.sqrt(sample_size))
        kiUpper = population + alpha_ninefive * (std/math.sqrt(sample_size))
        kiLowerTwo = population - alpha_ninenine * (std / math.sqrt(sample_size))
        kiUpperTwo = population + alpha_ninenine * (std / math.sqrt(sample_size))



    print("\nZ-Score: " + str(round(result, 2)))
    print("\n95%-Konfidenz Interval: [" + str(round(kiLower,4)) + ";" + str(round(kiUpper,4)) + "]")
    print("99%-Konfidenz Interval: [" + str(round(kiLowerTwo,4)) + ";" + str(round(kiUpperTwo,4)) + "]")

    print("\nSignifikanz-Niveau gegeben? (Y/N): ")
    sig_given = input()
    if sig_given == "Y" or sig_given == "y":
        print("(α) Signifikanz Niveau:")
        alpha = float(input().replace(",", "."))
        print("\nZ-Kritisch:")
        print("Linksseitig: " +  str(scipy.stats.norm.ppf(round(alpha, 2))))
        print("Rechtsseitig: " + str(scipy.stats.norm.ppf(1 - round(alpha, 2))))
        print("Zweiseitig: " + str(scipy.stats.norm.ppf(1 - round(alpha, 2) / 2)))

    input("\nEingabe Taste drücken um fortzufahren...")


def t_test():
    print("\nT-Test")

    print("(x̄) Mittelwert der Stichprobe:")
    sample = float(input().replace(",", "."))

    print("(μ) Mittelwert der Population:")
    population = float(input().replace(",", "."))

    print("(s) Standard-Abweichung:")
    std = float(input().replace(",", "."))

    print("(n) Umfang der Stichprobe:")
    sample_size = int(input())

    df = sample_size - 1

    t_nineFive = scipy.stats.t.ppf(1 - .05 / 2, df)
    t_nineNine = scipy.stats.t.ppf(1 - .01 / 2, df)
    kiLower = population - t_nineFive * (std / math.sqrt(sample_size))
    kiUpper = population + t_nineFive * (std / math.sqrt(sample_size))
    kiLowerTwo = population - t_nineNine * (std / math.sqrt(sample_size))
    kiUpperTwo = population + t_nineNine * (std / math.sqrt(sample_size))

    t_value = result = (sample - population)/(std/math.sqrt(sample_size))

    print("T-Value: " + str(t_value))
    print("\n95%-Konfidenz Interval: [" + str(round(kiLower, 4)) + ";" + str(round(kiUpper, 4)) + "]")
    print("99%-Konfidenz Interval: [" + str(round(kiLowerTwo, 4)) + ";" + str(round(kiUpperTwo, 4)) + "]")

    print("\nSignifikanz-Niveau gegeben? (Y/N): ")
    sig_given = input()
    if sig_given == "Y" or sig_given == "y":
        print("(α) Signifikanz Niveau:")
        alpha = float(input().replace(",", "."))
        t_crit_left = scipy.stats.t.ppf(alpha, df)
        t_crit_right = scipy.stats.t.ppf(1-alpha, df)
        t_crit_both = scipy.stats.t.ppf(1 - alpha/2, df)
        print("\nT-Kritisch")
        print("Linksseitig: " + str(t_crit_left))
        print("Rechtsseitig: " + str(t_crit_right))
        print("Zweiseitig: " + str(t_crit_both))

    input("\nEingabe Taste drücken um fortzufahren...")


def t_test_unabhaengig():
    print("t-Test Abhängige Stichproben")

    print("(x̄1) Mittelwert der Stichprobe 1:")
    sample = float(input().replace(",", "."))

    print("(x̄2) Mittelwert der Stichprobe 2:")
    sample2 = float(input().replace(",", "."))

    print("(s1-s2) Standardfehler der Differenz:")
    std = float(input().replace(",", "."))

    print("(n) Umfang der Stichprobe:")
    sample_size = int(input())

    print("(α) Signifikanz Niveau:")
    alpha = float(input().replace(",", "."))

    df = sample_size-2

    MD = sample - sample2

    print("(x̄1 - x̄2) Mittlere differenz: " + str(round(MD,2)))

    kiLower = MD - scipy.stats.t.ppf(1 - alpha / 2, df) * std
    kiUpper = MD + scipy.stats.t.ppf(1 - alpha / 2, df) * std

    print("\nKonfidenz Interval: [" + str(round(kiLower, 4)) + ";" + str(round(kiUpper, 4)) + "]")

    print("T-Value: " + str(round(MD/std, 4)))
    print("df: " + str(df))
    t_crit_left = scipy.stats.t.ppf(alpha, df)
    t_crit_right = scipy.stats.t.ppf(1 - alpha, df)
    t_crit_both = scipy.stats.t.ppf(1 - alpha / 2, df)
    print("\nT-Kritisch")
    print("Linksseitig: " + str(t_crit_left))
    print("Rechtsseitig: " + str(t_crit_right))
    print("Zweiseitig: " + str(t_crit_both))

    input("\nEingabe Taste drücken um fortzufahren...")


def t_test_abhaengig():
    print("t-Test Abhängige Stichproben")

    print("(x̄D) Mittelwert der Stichprobe:")
    sample = float(input().replace(",", "."))

    print("(μ) Mittelwert der Population:")
    sample2 = float(input().replace(",", "."))

    print("(sx̄D) Standardfehler der Differenz:")
    std = float(input().replace(",", "."))

    print("(n) Umfang der Stichprobe:")
    sample_size = int(input())

    print("(α) Signifikanz Niveau:")
    alpha = float(input().replace(",", "."))

    df = sample_size-1

    print("T-Value: " + str(round((sample-sample2)/std, 4)))
    print("df: " + str(df))
    t_crit_left = scipy.stats.t.ppf(alpha, df)
    t_crit_right = scipy.stats.t.ppf(1 - alpha, df)
    t_crit_both = scipy.stats.t.ppf(1 - alpha / 2, df)
    print("\nT-Kritisch")
    print("Linksseitig: " + str(t_crit_left))
    print("Rechtsseitig: " + str(t_crit_right))
    print("Zweiseitig: " + str(t_crit_both))

    input("\nEingabe Taste drücken um fortzufahren...")

def chi_squared():
    print("\nChi Quadrat Konfidenz Interval")

    print("(n) Umfang der Stichprobe:")
    sample_size = int(input())

    print("(s) Standard-Abweichung:")
    std = float(input().replace(",", "."))

    print("(α) Signifikanz Niveau:")
    alpha = float(input().replace(",", "."))

    df = sample_size - 1

    chiLeft = scipy.stats.chi2.ppf(alpha/2, df)
    chiRight = scipy.stats.chi2.ppf(1 - (alpha / 2), df)

    kiLower = (df * (std**2)) / chiLeft
    kiUpper = (df * (std**2)) / chiRight

    print("\nKonfidenz Interval: [" + str(round(kiLower, 4)) + ";" + str(round(kiUpper, 4)) + "]")

    input("\nEingabe Taste drücken um fortzufahren...")



def tschebyscheff():
    print("Tschebyscheff")

    print("Obere oder Untere Grenze? (O/U): ")
    border = input()

    print("E(x) Erwartungswert eingeben: ")
    expected = float(input().replace(",", "."))

    print("Var(x) Varianz Eingeben: ")
    var = float(input().replace(",", "."))

    if border == "U" or border == "u":
        print("Untere Grenze Eingeben: ")
        borderUnder = float(input().replace(",", "."))
        k = abs(expected - borderUnder)
        print("P(|X-E(x)| < UG) = " + str(round(1 - (var / (k ** 2)), 2)))
    if border == "O" or border == "o":
        print("Obere Grenze Eingeben: ")
        borderUpper = float(input().replace(",", "."))
        k = abs(expected - borderUpper)
        print("P(|X-E(x)| >= OG) = " + str(round(var / (k ** 2), 4)))

    input("\nEingabe Taste drücken um fortzufahren...")



def metric_calc():
    print("\n4-Felder Taffel auswertung")

    print("\n(Links Oben) True Positive Anzahl:")
    TP = int(input())
    print("(Links Unten)False Positive Anzahl:")
    FP = int(input())
    print("(Rechts Unten) True Negative Anzahl:")
    TN = int(input())
    print("(Oben Rechts)False Negative Anzahl:")
    FN = int(input())

    P = TP + FP
    N = TN + FN

    Praevalenz = P / (P + N)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = 1 - FPR
    FNR = 1 - TNR
    FDR = FP / (TP + FP)
    PPV = TP / (TP + FP)
    Accuracy = (TP + TN) / (P + N)
    F1_Score = 2 * (PPV * TPR) / (PPV + TPR)
    Youden = TPR + TNR - 1
    IoU = TP / (TP + FP + FN)

    print("\nErgebnisse:\n\nPrävalenz: " + str(round(Praevalenz)))
    print("Recall/Sensitivität: " + str(round(TPR, 2)))
    print("False Positive Rate: "+ str(round(FPR, 2)))
    print("Spezifität: " + str(round(TNR, 2)))
    print("False Negativ Rate: " + str(round(FNR, 2)))
    print("False Discovery Rate(FDR): " + str(round(FDR, 2)))
    print("Precision: " + str(round(PPV, 2)))
    print("Accuracy: " + str(round(Accuracy, 2)))
    print("F1-Score: " + str(round(F1_Score, 2)))
    print("Youden Index: " + str(round(Youden, 2)))
    print("Intersection over Union: " + str(round(IoU, 2)))

    input("\nEingabe Taste drücken um fortzufahren...")


def contigenz():
    print("\nPearson Contigenz")

    print("(X^2) Quadratische Contingenz:")
    x = float(input().replace(",", "."))

    print("(n) Gesamtanzahl: ")
    sample_size = int(input())

    print("k min(zeilen, spalten): ")
    k = int(input())

    pearson_c = math.sqrt(x / (x + sample_size))
    c_corr = pearson_c * math.sqrt(k / (k - 1))

    print("\nPearsons-C: " + str(round(pearson_c, 2)))
    print("Quadratische Kontingenz: " + str(round(c_corr, 2)) + "\n")

    zusammenhang = ""
    x = abs(c_corr)

    if x < 0.3: zusammenhang = "schwachem"
    elif 0.3 <= x < 0.4: zusammenhang = "mittlerem"
    elif 0.4 <= x < 0.7: zusammenhang = "starkem"
    elif 0.7<= x < 1: zusammenhang = "direktem"

    if c_corr < 0:
        print("=> Variablen befinden sich im " + zusammenhang + " negativem Zusammenhang.")
    else:
        print("=> Variablen befinden sich im " + zusammenhang + " positiven Zusammenhang.")

    input("\nEingabe Taste drücken um fortzufahren...")


def display_menu():
    print("\n---- Probability Calculation Menu ----")

    print("\n--- Stochastik ---")
    print("1. Hypergeometrische Wahrscheinlichkeit")
    print("2. Poisson Wahrscheinlichkeit")
    print("3. Binomial Wahrscheinlichkeit")

    print("\n--- Hypothesen Tests ---")
    print("4. Z-Test (n >= 30)")
    print("5. T-Test (n < 30)")
    print("6. T-Test unabhängige Stichproben")
    print("7. T-Test abhängige Stichproben")
    print("8. Chi-Quadrat Konfidenzinterval")
    print("9. Tschebycheff-Ungleichung (Keine bestimmte Verteilung)")

    print("\n--- Verschiedenes ---")
    print("10. Confusion Matrix")
    print("11. Kontigenz Koeffizient nach Pearson")

    print("\n12. Exit")


# Main program
while True:
    display_menu()
    choice = input("\nEnter your choice (1-12): ")

    if choice == "1":
        calculate_hypergeometric_probability()
    elif choice == "2":
        calculate_poison_probability()
    elif choice == "3":
        calculate_binomial_probability()
    elif choice == "4":
        z_zest()
    elif choice == "5":
        t_test()
    elif choice == "6":
        t_test_unabhaengig()
    elif choice == "7":
        t_test_abhaengig()
    elif choice == "8":
        chi_squared()
    elif choice == "9":
        tschebyscheff()
    elif choice == "10":
        metric_calc()
    elif choice == "11":
        contigenz()
    elif choice == "12":
        print("Exiting the program...")
        break
    else:
        print("Invalid choice. Please try again.\n")