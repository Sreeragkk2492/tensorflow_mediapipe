import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

class ControlsOverlay extends StatelessWidget {
  ControlsOverlay({
    required this.controller,
    required this.isVisible,
    required this.onHideControls,
  });

  final VideoPlayerController controller;
  final bool isVisible;
  final VoidCallback onHideControls;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: <Widget>[
       
        Align(
          alignment: Alignment.bottomCenter,
          child: Container(
            color: Colors.black26,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  icon: Icon(
                    controller.value.isPlaying ? Icons.pause : Icons.play_arrow,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    controller.value.isPlaying
                        ? controller.pause()
                        : controller.play();
                  },
                ),
                IconButton(
                  icon: Icon(
                    Icons.stop,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    controller.seekTo(Duration.zero);
                    controller.pause();
                  },
                ),
                Expanded(
                  child: VideoProgressIndicator(
                    controller,
                    allowScrubbing: true,
                    colors: VideoProgressColors(
                      playedColor: Colors.red,
                      backgroundColor: Colors.grey,
                      bufferedColor: Colors.white,
                    ),
                  ),
                ),
                IconButton(
                  icon: Icon(
                    Icons.fullscreen,
                    color: Colors.white,
                  ),
                  onPressed: () {
                    // Fullscreen logic here
                  },
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
