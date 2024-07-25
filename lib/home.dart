import 'dart:async';


import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:tensorflow_mediapipe/test.dart';
import 'package:tensorflow_mediapipe/widgets/controls.dart';
import 'package:video_player/video_player.dart';

class VideoPlayerScreen extends StatefulWidget {
  VideoPlayerScreen({super.key});

  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  Timer? _hideControlsTimer;

  bool _isControlsVisible = false;

  void _showControls() {
    setState(() {
      _isControlsVisible = true;
    });
    _hideControlsAfterDelay();
  }

  void _hideControlsAfterDelay() {
    _hideControlsTimer?.cancel();
    _hideControlsTimer = Timer(Duration(seconds: 3), () {
      setState(() {
        _isControlsVisible = false;
      });
    });
  }

  final videoPlayerController = Get.put(GestureVideoController());

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video Player with Gesture Control'),
      ),
      body: Obx(
        () => videoPlayerController.isVideoInitialized.value
            ? Column(
                children: [
                  MouseRegion(
                    onHover: (_) => _showControls(),
                    child: GestureDetector(
                      onTap: () => _showControls(),
                      child: AspectRatio(
                        aspectRatio: videoPlayerController
                            .videoController.value.aspectRatio,
                        child:
                            Stack(alignment: Alignment.bottomCenter, children: [
                          VideoPlayer(videoPlayerController.videoController),
                          AnimatedOpacity(
                            opacity: _isControlsVisible ? 1.0 : 0.0,
                            duration: Duration(milliseconds: 300),
                            child: ControlsOverlay(
                              controller: videoPlayerController.videoController,
                              isVisible: _isControlsVisible,
                              onHideControls: _hideControlsAfterDelay,
                            ),
                          ),

                          // Positioned(
                          //   bottom: 10,
                          //   left: 10,
                          //   child: IconButton(
                          //     icon: Obx(
                          //       () => Icon(
                          //         videoPlayerController.isVideoPlaying.value
                          //             ? Icons.pause
                          //             : Icons.play_arrow,
                          //         color: Colors.black,
                          //         size: 30.0,
                          //       ),
                          //     ),
                          //     onPressed: () {
                          //       videoPlayerController.toggleVideoPlayback();
                          //     },
                          //   ),
                          // ),
                        ]),
                      ),
                    ),
                  ),
                  // Text(
                  //  // videoPlayerController.label.value,

                  //   style: TextStyle(fontSize: 20),
                  // )
                  // SizedBox(
                  //   height: 70,
                  // ),
                  // AvatarGlow(
                  //   glowColor: Colors.blue,
                  //   duration: const Duration(milliseconds: 2000),
                  //   repeat: true,
                  //   glowCount: 2,
                  //   startDelay: const Duration(milliseconds: 100),
                  //   animate: videoPlayerController.isTtsActive.value ||
                  //       videoPlayerController.isSstActive.value,
                  //   child: CircleAvatar(
                  //     backgroundColor: Colors.blue,
                  //     radius: 50.0,
                  //     child: Icon(
                  //         videoPlayerController.isTtsActive.value
                  //             ? Icons.person
                  //             : Icons.mic,
                  //         size: 50,
                  //         color: Colors.white),
                  //   ),
                  // ),
                  // SizedBox(
                  //   height: 20,
                  // ),
                  // Obx(
                  //   () => Text(
                  //     videoPlayerController.recognizedWords.value,
                  //     style: TextStyle(fontSize: 20),
                  //   ),
                  // )
                ],
              )
            : const Center(
                child: CircularProgressIndicator(),
              ),
      ),
      // floatingActionButton: FloatingActionButton(
      //   backgroundColor: Colors.blue,
      //   onPressed: videoPlayerController.toggleVideoPlayback,
      //   child: Obx(
      //     () => Icon(
      //       videoPlayerController.isVideoPlaying.value
      //           ? Icons.pause
      //           : Icons.play_arrow,
      //     ),
      //   ),
      // ),
    );
  }
}
