import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';

/// Evaluates compact XGBoost decision tree models exported by export_mobile.py.
/// Runs completely offline on the device without any external native library.
class XgbNutritionModel {
  final int numClass;
  final List<String> features;
  final List<int> treeClass;
  final List<Map<String, dynamic>> trees;

  XgbNutritionModel({
    required this.numClass,
    required this.features,
    required this.treeClass,
    required this.trees,
  });

  static Future<XgbNutritionModel> loadFromAsset(String assetPath) async {
    final raw = await rootBundle.loadString(assetPath);
    final data = jsonDecode(raw) as Map<String, dynamic>;
    return XgbNutritionModel(
      numClass: data['num_class'] as int,
      features: List<String>.from(data['features']),
      treeClass: List<int>.from(data['tree_class']),
      trees: List<Map<String, dynamic>>.from(data['trees']),
    );
  }

  /// Evaluates nutrition values per 100g.
  /// [values] should match [features]:
  /// [energy-kcal, sugars, saturated-fat, salt, fiber, proteins].
  List<double> predictProba(List<double?> values) {
    final margins = List<double>.filled(numClass, 0.0);

    for (int i = 0; i < trees.length; i++) {
      final t = trees[i];
      final cls = treeClass[i];
      final left = t['l'] as List;
      final right = t['r'] as List;
      final feats = t['f'] as List;
      final conds = t['c'] as List;
      final defLeft = t['d'] as List;

      int n = 0;
      while (left[n] != -1) {
        final featIdx = feats[n] as int;
        final v = featIdx < values.length ? values[featIdx] : null;
        if (v == null || v.isNaN) {
          n = defLeft[n] == 1 ? (left[n] as int) : (right[n] as int);
        } else {
          final cond = (conds[n] as num).toDouble();
          n = v < cond ? (left[n] as int) : (right[n] as int);
        }
      }
      margins[cls] += (conds[n] as num).toDouble();
    }

    final maxMargin = margins.reduce(math.max);
    final exps = margins.map((m) => math.exp(m - maxMargin)).toList();
    final sumExp = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExp).toList();
  }

  int predict(List<double?> values) {
    final proba = predictProba(values);
    int best = 0;
    for (int i = 1; i < proba.length; i++) {
      if (proba[i] > proba[best]) best = i;
    }
    return best; // 0=A, 1=B, 2=C, 3=D, 4=E
  }
}
