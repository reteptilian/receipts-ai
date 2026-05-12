import 'package:cunning_document_scanner/cunning_document_scanner.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

void main() {
  runApp(const ReceiptTrackerApp());
}

class ReceiptTrackerApp extends StatelessWidget {
  const ReceiptTrackerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Receipt Tracker',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff16645a)),
        useMaterial3: true,
      ),
      home: const ReceiptScanScreen(),
    );
  }
}

class ReceiptScanScreen extends StatefulWidget {
  const ReceiptScanScreen({super.key});

  @override
  State<ReceiptScanScreen> createState() => _ReceiptScanScreenState();
}

class _ReceiptScanScreenState extends State<ReceiptScanScreen> {
  var _isScanning = false;
  var _status = _isIos
      ? 'Ready to scan a receipt.'
      : 'Receipt scanning is only wired up for iOS right now.';
  var _ocrText = '';
  var _scannedPageCount = 0;

  static bool get _isIos =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.iOS;

  Future<void> _scanReceipt() async {
    if (!_isIos || _isScanning) {
      return;
    }

    setState(() {
      _isScanning = true;
      _status = 'Opening the iOS document scanner...';
      _ocrText = '';
      _scannedPageCount = 0;
    });

    try {
      final imagePaths = await CunningDocumentScanner.getPictures(
        noOfPages: 4,
        isGalleryImportAllowed: false,
        iosScannerOptions: const IosScannerOptions(
          imageFormat: IosImageFormat.jpg,
          jpgCompressionQuality: 0.9,
        ),
      );

      if (!mounted) {
        return;
      }

      if (imagePaths == null || imagePaths.isEmpty) {
        setState(() {
          _status = 'Scan cancelled.';
          _isScanning = false;
        });
        return;
      }

      setState(() {
        _status = 'Running OCR on ${imagePaths.length} scanned page(s)...';
        _scannedPageCount = imagePaths.length;
      });

      final ocrText = await _recognizeReceiptText(imagePaths);
      debugPrint('Receipt OCR result:\n$ocrText');

      if (!mounted) {
        return;
      }

      setState(() {
        _ocrText = ocrText;
        _status = ocrText.trim().isEmpty
            ? 'OCR completed, but no text was found.'
            : 'OCR completed. Results printed to the debug console.';
      });
    } catch (error, stackTrace) {
      debugPrint('Receipt scan/OCR failed: $error');
      debugPrintStack(stackTrace: stackTrace);

      if (!mounted) {
        return;
      }

      setState(() {
        _status = 'Scan/OCR failed: $error';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isScanning = false;
        });
      }
    }
  }

  Future<String> _recognizeReceiptText(List<String> imagePaths) async {
    final recognizer = TextRecognizer(script: TextRecognitionScript.latin);
    final pageOutputs = <String>[];

    try {
      for (final (index, imagePath) in imagePaths.indexed) {
        final inputImage = InputImage.fromFilePath(imagePath);
        final recognizedText = await recognizer.processImage(inputImage);
        final lines = recognizedText.blocks
            .expand((block) => block.lines)
            .map((line) => line.text.trim())
            .where((line) => line.isNotEmpty)
            .toList(growable: false);
        final pageText = lines.isEmpty
            ? recognizedText.text.trim()
            : lines.join('\n');

        pageOutputs.add('--- Page ${index + 1} ---\n$pageText');
      }
    } finally {
      await recognizer.close();
    }

    return pageOutputs.join('\n\n');
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Receipt Tracker'),
        backgroundColor: colorScheme.surface,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Receipt Scan',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 8),
              Text(_status, style: Theme.of(context).textTheme.bodyLarge),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _isIos && !_isScanning ? _scanReceipt : null,
                icon: _isScanning
                    ? const SizedBox.square(
                        dimension: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.document_scanner_outlined),
                label: Text(_isScanning ? 'Scanning...' : 'Scan receipt'),
              ),
              const SizedBox(height: 16),
              if (_scannedPageCount > 0)
                Text(
                  'Scanned pages: $_scannedPageCount',
                  style: Theme.of(context).textTheme.labelLarge,
                ),
              const SizedBox(height: 8),
              Expanded(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    border: Border.all(color: colorScheme.outlineVariant),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(16),
                    child: SelectableText(
                      _ocrText.isEmpty
                          ? 'OCR results will appear here after a scan.'
                          : _ocrText,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontFamily: 'monospace',
                        height: 1.35,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
