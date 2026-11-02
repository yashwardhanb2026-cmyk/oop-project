import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';

/// Evaluates TF-IDF + Logistic Regression ingredients model exported by export_mobile.py.
/// Runs completely offline on device.
class TextNutritionModel {
  final List<double> bias;
  final Map<String, List<double>> vocab;

  TextNutritionModel({
    required this.bias,
    required this.vocab,
  });

  static Future<TextNutritionModel> loadFromAsset(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final data = jsonDecode(raw) as Map<String, dynamic>;
    final biasList = (data['bias'] as List).map((b) => (b as num).toDouble()).toList();
    final rawVocab = data['vocab'] as Map<String, dynamic>;
    final vocabMap = <String, List<double>>{};
    rawVocab.forEach((key, val) {
      vocabMap[key] = (val as List).map((v) => (v as num).toDouble()).toList();
    });
    return TextNutritionModel(bias: biasList, vocab: vocabMap);
  }

  List<double> predictProba(String text) {
    final reg = RegExp(r'[a-z]{2,}');
    final matches = reg.allMatches(text.toLowerCase()).map((m) => m.group(0)!).toList();
    final terms = List<String>.from(matches);
    for (int i = 0; i < matches.length - 1; i++) {
      terms.add('${matches[i]} ${matches[i + 1]}');
    }

    final counts = <String, int>{};
    for (final t in terms) {
      counts[t] = (counts[t] ?? 0) + 1;
    }

    final feats = <String, double>{};
    counts.forEach((t, c) {
      if (vocab.containsKey(t)) {
        final idf = vocab[t]![0];
        feats[t] = (1.0 + math.log(c)) * idf;
      }
    });

    final logits = List<double>.from(bias);
    double sumSquares = 0.0;
    feats.forEach((_, v) {
      sumSquares += v * v;
    });
    final norm = math.sqrt(sumSquares);

    if (norm > 0) {
      feats.forEach((t, v) {
        final weights = vocab[t]!.sublist(1);
        final factor = v / norm;
        for (int c = 0; c < logits.length; c++) {
          logits[c] += factor * weights[c];
        }
      });
    }

    final maxLogit = logits.reduce(math.max);
    final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
    final sumExp = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExp).toList();
  }

  int predict(String text) {
    final proba = predictProba(text);
    int best = 0;
    for (int i = 1; i < proba.length; i++) {
      if (proba[i] > proba[best]) best = i;
    }
    return best; // 0=A, 1=B, 2=C, 3=D, 4=E
  }
}
