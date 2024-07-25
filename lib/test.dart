import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:awesome_dialog/awesome_dialog.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:get/get.dart';
import 'package:camera/camera.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
import 'package:video_player/video_player.dart';
import 'package:image/image.dart' as img_lib;

class GestureVideoController extends GetxController {
  late CameraController cameraController;
  late VideoPlayerController videoController;
  CameraImage? cameraImage;
  bool isDetecting = false;
  bool isCameraInitialized = false;
  RxBool isVideoInitialized = false.obs;
  RxBool isVideoPlaying = true.obs;
  FlutterTts flutterTts = FlutterTts();
  final SpeechToText speechToText = SpeechToText();
  late Interpreter interpreter;
  late ImageProcessor imageProcessor;
  late TensorImage inputImage;
  late List<Object> inputs;

  Map<int, Object> outputs = {};
  TensorBuffer outputLocations = TensorBufferFloat([]);

  Stopwatch s = Stopwatch();

  int frameNo = 0;

  @override
  void onInit() {
    super.onInit();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      showInitializationDialog();
    });
    loadModel();
    initializeVideoPlayer();
  }

  loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/handlandmark.tflite');
      print("Mediapipe model loaded successfully");
      print("Input shape: ${interpreter.getInputTensor(0).shape}");
      print("Output shape: ${interpreter.getOutputTensor(0).shape}");
    } catch (e) {
      print('Failed to load Mediapipe model: $e');
      print('Stack trace: ${StackTrace.current}');
    }
  }

  showInitializationDialog() async {
    AwesomeDialog dialog = AwesomeDialog(
      context: Get.context!,
      animType: AnimType.bottomSlide,
      dialogType: DialogType.info,
      body: Center(
        child: Column(
          children: <Widget>[
            Text(
              'AI assistant',
              style: TextStyle(fontSize: 22),
            ),
            SizedBox(height: 10),
            Text(
              'Do you want to continue with the assistant?',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
      btnOkOnPress: () async {
        await initializeCamera();
        {
          Get.back(); // Close dialog
        }
      },
      btnCancelOnPress: () {
        Get.back(); // Close dialog
      },
    );

    dialog.show();
  }

  Future<bool> initializeCamera() async {
    List<CameraDescription> cameras = await availableCameras();

    CameraDescription? frontCamera;
    for (CameraDescription camera in cameras) {
      if (camera.lensDirection == CameraLensDirection.front) {
        frontCamera = camera;
        break;
      }
    }

    cameraController = CameraController(
      frontCamera!,
      ResolutionPreset.high,
    );

    try {
      await cameraController.initialize();
      isCameraInitialized = true;
      await Future.delayed(const Duration(seconds: 2));

      if (cameraController.value.isInitialized) {
        cameraController.startImageStream((imageStream) async {
          if (!isDetecting) {
            isDetecting = true;
            runModelOnFrame(imageStream);
          }
        });
      }

      return true;
    } catch (e) {
      print("Error initializing camera: $e");
      return false;
    }
  }

  List<double> preprocessImage(CameraImage image) {
    // Convert YUV420 to RGB
    img_lib.Image img = convertYUV420ToImage(image);

    // Resize the image to 256x256 (typical input size for MediaPipe hand landmark model)
    img_lib.Image resizedImg = img_lib.copyResize(img, width: 256, height: 256);

    // Convert the image to a list of normalized pixel values
    List<double> input = imageToByteListFloat32(resizedImg);

    return input;
  }

  img_lib.Image convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

    final image = img_lib.Image(width, height);

    for (int w = 0; w < width; w++) {
      for (int h = 0; h < height; h++) {
        final int uvIndex =
            uvPixelStride * (w / 2).floor() + uvRowStride * (h / 2).floor();
        final int index = h * width + w;

        final y = cameraImage.planes[0].bytes[index];
        final u = cameraImage.planes[1].bytes[uvIndex];
        final v = cameraImage.planes[2].bytes[uvIndex];

        image.data[index] = yuv2rgb(y, u, v);
      }
    }
    return image;
  }

  int yuv2rgb(int y, int u, int v) {
    // Convert YUV to RGB
    int r = (y + v * 1436 / 1024 - 179).round();
    int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
    int b = (y + u * 1814 / 1024 - 227).round();

    // Clipping RGB values to be inside [0, 255]
    r = r.clamp(0, 255);
    g = g.clamp(0, 255);
    b = b.clamp(0, 255);

    return 0xff000000 |
        ((b << 16) & 0xff0000) |
        ((g << 8) & 0xff00) |
        (r & 0xff);
  }

  List<double> imageToByteListFloat32(img_lib.Image image) {
    var convertedBytes = Float32List(1 * 256 * 256 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < 256; i++) {
      for (var j = 0; j < 256; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (img_lib.getRed(pixel) - 127.5) / 127.5;
        buffer[pixelIndex++] = (img_lib.getGreen(pixel) - 127.5) / 127.5;
        buffer[pixelIndex++] = (img_lib.getBlue(pixel) - 127.5) / 127.5;
      }
    }
    return convertedBytes.toList();
  }

  void runModelOnFrame(CameraImage image) async {
    if (!isDetecting) {
      isDetecting = true;
      try {
        print('preprocessing images');
        var input = preprocessImage(image);

        var output = List.filled(63, 0.0).reshape([1, 63]);
        interpreter.run(input, output);
        print('detecting gestures...');
        detectGesture(output[0]);
        print('output: ${output.toString()}');
      } catch (e) {
        print("Error running model: $e");
      } finally {
        isDetecting = false;
      }
    }
  }

  void detectGesture(List<double> landmarks) {
    // Example: Detect if the hand is open
    bool isHandOpen = isHandOpenGesture(landmarks);

    // Example: Detect if the hand is raised
    bool isHandRaised = isHandRaisedGesture(landmarks);

    if (isHandRaised) {
      if (isVideoPlaying.value) {
        videoController.play();
        isVideoPlaying.value = false;
      }
    } else if (isHandOpen) {
      if (!isVideoPlaying.value) {
        videoController.pause();
        isVideoPlaying.value = true;
      }
    }
  }

  bool isHandOpenGesture(List<double> landmarks) {
    // Example implementation - you might need to adjust this based on your needs
    double thumbTip = landmarks[4 * 3 + 1]; // y-coordinate of thumb tip
    double indexTip = landmarks[8 * 3 + 1]; // y-coordinate of index finger tip
    double pinkyTip = landmarks[20 * 3 + 1]; // y-coordinate of pinky tip

    // If finger tips are significantly higher than the wrist, consider the hand open
    return (thumbTip < landmarks[0 * 3 + 1] &&
        indexTip < landmarks[0 * 3 + 1] &&
        pinkyTip < landmarks[0 * 3 + 1]);
  }

  bool isHandRaisedGesture(List<double> landmarks) {
    // Example implementation - you might need to adjust this based on your needs
    double wristY = landmarks[0 * 3 + 1];

    // If the wrist is in the upper half of the image, consider the hand raised
    return wristY < 0.5;
  }

  Future<void> initializeVideoPlayer() async {
    videoController = VideoPlayerController.asset('assets/video1.mp4');
    await videoController.initialize();
    isVideoInitialized.value = true;
    videoController.play();
    videoController.setLooping(true);
  }

  @override
  void onClose() {
    videoController.dispose();
    cameraController.dispose();
    interpreter.close();
    super.onClose();
  }
}
