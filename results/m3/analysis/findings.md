# Findings (test set, n = 3,080; 95% CIs from 2,000 paired bootstrap resamples; McNemar exact, Holm-adjusted)

## frozen (MiniLM+clf): C1 raw Macro-F1 0.894 (95% CI 0.882 to 0.904)
- original (no contact sentence): Macro-F1 0.916, +2.2 pp vs raw (95% CI +1.4 pp to +3.0 pp); accuracy +2.2 pp; McNemar p < 0.001: a detectable difference at n = 3,080.
- delete: Macro-F1 0.903, +0.9 pp vs raw (95% CI +0.1 pp to +1.7 pp); accuracy +0.9 pp; McNemar p = 0.06 (Holm): not a detectable difference at n = 3,080.
- generic [REDACTED]: Macro-F1 0.857, -3.7 pp vs raw (95% CI -4.8 pp to -2.7 pp); accuracy -4.2 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- typed [TYPE]: Macro-F1 0.901, +0.6 pp vs raw (95% CI -0.1 pp to +1.5 pp); accuracy +0.6 pp; McNemar p = 0.15 (Holm): not a detectable difference at n = 3,080.
- delete (detector): Macro-F1 0.895, +0.1 pp vs raw (95% CI -0.8 pp to +0.9 pp); accuracy +0.1 pp; McNemar p = 0.88 (Holm): not a detectable difference at n = 3,080.
- generic (detector): Macro-F1 0.846, -4.8 pp vs raw (95% CI -6.0 pp to -3.9 pp); accuracy -5.5 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- typed (detector): Macro-F1 0.889, -0.5 pp vs raw (95% CI -1.4 pp to +0.4 pp); accuracy -0.6 pp; McNemar p = 0.49 (Holm): not a detectable difference at n = 3,080.
- Highest test Macro-F1 among methods (descriptive): delete (+0.9 pp vs raw). Method chosen on validation: generic.
- Cost of detector errors for generic: -1.13 pp Macro-F1 (D vs C).

## retrained (MiniLM+clf): C1 raw Macro-F1 0.894 (95% CI 0.882 to 0.904)
- original (no contact sentence): Macro-F1 0.927, +3.2 pp vs raw (95% CI +2.4 pp to +4.1 pp); accuracy +3.3 pp; McNemar p < 0.001: a detectable difference at n = 3,080.
- delete: Macro-F1 0.919, +2.4 pp vs raw (95% CI +1.7 pp to +3.2 pp); accuracy +2.5 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- generic [REDACTED]: Macro-F1 0.916, +2.2 pp vs raw (95% CI +1.5 pp to +3.0 pp); accuracy +2.2 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- typed [TYPE]: Macro-F1 0.914, +1.9 pp vs raw (95% CI +1.3 pp to +2.7 pp); accuracy +2.0 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- delete (detector): Macro-F1 0.910, +1.6 pp vs raw (95% CI +0.8 pp to +2.4 pp); accuracy +1.6 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- generic (detector): Macro-F1 0.904, +0.9 pp vs raw (95% CI +0.2 pp to +1.8 pp); accuracy +1.0 pp; McNemar p = 0.04 (Holm): a detectable difference at n = 3,080.
- typed (detector): Macro-F1 0.902, +0.8 pp vs raw (95% CI +0.0 pp to +1.6 pp); accuracy +0.8 pp; McNemar p = 0.06 (Holm): not a detectable difference at n = 3,080.
- Highest test Macro-F1 among methods (descriptive): delete (+2.4 pp vs raw). Method chosen on validation: generic.
- Cost of detector errors for generic: -1.26 pp Macro-F1 (D vs C).

## frozen_tfidf (TF-IDF+SVM): C1 raw Macro-F1 0.880 (95% CI 0.866 to 0.890)
- original (no contact sentence): Macro-F1 0.876, -0.4 pp vs raw (95% CI -1.0 pp to +0.2 pp); accuracy -0.4 pp; McNemar p = 0.24: not a detectable difference at n = 3,080.
- delete: Macro-F1 0.878, -0.2 pp vs raw (95% CI -0.7 pp to +0.3 pp); accuracy -0.2 pp; McNemar p = 0.90 (Holm): not a detectable difference at n = 3,080.
- generic [REDACTED]: Macro-F1 0.879, -0.1 pp vs raw (95% CI -0.6 pp to +0.4 pp); accuracy -0.1 pp; McNemar p = 0.90 (Holm): not a detectable difference at n = 3,080.
- typed [TYPE]: Macro-F1 0.872, -0.8 pp vs raw (95% CI -1.3 pp to -0.2 pp); accuracy -1.0 pp; McNemar p = 0.002 (Holm): a detectable difference at n = 3,080.
- delete (detector): Macro-F1 0.875, -0.5 pp vs raw (95% CI -1.0 pp to +0.1 pp); accuracy -0.5 pp; McNemar p = 0.20 (Holm): not a detectable difference at n = 3,080.
- generic (detector): Macro-F1 0.876, -0.4 pp vs raw (95% CI -0.9 pp to +0.1 pp); accuracy -0.4 pp; McNemar p = 0.20 (Holm): not a detectable difference at n = 3,080.
- typed (detector): Macro-F1 0.869, -1.1 pp vs raw (95% CI -1.6 pp to -0.5 pp); accuracy -1.3 pp; McNemar p < 0.001 (Holm): a detectable difference at n = 3,080.
- Highest test Macro-F1 among methods (descriptive): generic (-0.1 pp vs raw). Method chosen on validation: generic.
- Cost of detector errors for generic: -0.32 pp Macro-F1 (D vs C).

## retrained_tfidf (TF-IDF+SVM): C1 raw Macro-F1 0.880 (95% CI 0.866 to 0.890)
- original (no contact sentence): Macro-F1 0.878, -0.2 pp vs raw (95% CI -0.9 pp to +0.5 pp); accuracy -0.2 pp; McNemar p = 0.66: not a detectable difference at n = 3,080.
- delete: Macro-F1 0.883, +0.3 pp vs raw (95% CI -0.4 pp to +1.0 pp); accuracy +0.3 pp; McNemar p = 1.00 (Holm): not a detectable difference at n = 3,080.
- generic [REDACTED]: Macro-F1 0.882, +0.2 pp vs raw (95% CI -0.5 pp to +0.9 pp); accuracy +0.2 pp; McNemar p = 1.00 (Holm): not a detectable difference at n = 3,080.
- typed [TYPE]: Macro-F1 0.882, +0.2 pp vs raw (95% CI -0.5 pp to +0.9 pp); accuracy +0.2 pp; McNemar p = 1.00 (Holm): not a detectable difference at n = 3,080.
- delete (detector): Macro-F1 0.879, -0.1 pp vs raw (95% CI -0.9 pp to +0.6 pp); accuracy -0.1 pp; McNemar p = 1.00 (Holm): not a detectable difference at n = 3,080.
- generic (detector): Macro-F1 0.878, -0.1 pp vs raw (95% CI -0.9 pp to +0.6 pp); accuracy -0.2 pp; McNemar p = 1.00 (Holm): not a detectable difference at n = 3,080.
- typed (detector): Macro-F1 0.878, -0.2 pp vs raw (95% CI -0.9 pp to +0.5 pp); accuracy -0.2 pp; McNemar p = 1.00 (Holm): not a detectable difference at n = 3,080.
- Highest test Macro-F1 among methods (descriptive): delete (+0.3 pp vs raw). Method chosen on validation: generic.
- Cost of detector errors for generic: -0.36 pp Macro-F1 (D vs C).

## Seeds (553, 554, 555)
- C1: mean Macro-F1 0.894 +/- 0.000 (sd).
- C3: mean Macro-F1 0.916 +/- 0.000 (sd).
