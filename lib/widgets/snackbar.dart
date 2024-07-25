
// import 'package:avatar_glow/avatar_glow.dart';
// import 'package:flutter/material.dart';
// import 'package:get/get.dart';
// import 'package:tensorflow_mediapipe/test.dart';

// class SnackbarContent extends StatelessWidget {
//   SnackbarContent({
//     super.key,
//   });

//   final  videoPlayerController = Get.put(GestureVideoController());

//   @override
//   Widget build(BuildContext context) {
//     return Container(
//       padding: EdgeInsets.all(20),
//       child: Row(
//         mainAxisSize: MainAxisSize.min,
//         children: [
//           Column(
//             children: [
//               SizedBox(
//                 height: 50,
//               ),
//               Obx(
//                 () => AvatarGlow(
//                   glowColor: videoPlayerController.isTtsActive.value
//                       ? Color.fromRGBO(6, 172, 255, 100)
//                       : Color.fromRGBO(6, 172, 255, 100),
//                   duration: const Duration(milliseconds: 2000),
//                   // repeat: true,
//                   glowCount: 2,
//                   startDelay: const Duration(milliseconds: 100),
//                   animate: videoPlayerController.isTtsActive.value , 
//                   child: CircleAvatar(
//                     backgroundColor: Colors.white,
//                     radius: 50.0,
//                     child: Obx(
//                       () => Icon(
//                         Icons.person,
//                         size: 50,
//                         color: videoPlayerController.isTtsActive.value
//                             ? Color.fromRGBO(6, 172, 255, 100)
//                             : Color.fromRGBO(6, 172, 255, 100),
//                       ),
//                     ),
//                   ),
//                 ),
//               ),
//             ],
//           ),
//           SizedBox(
//             width: 60,
//           ),
//           Column(children: [
//             SizedBox(
//               height: 50,
//             ),
//             Obx(
//               () => AvatarGlow(
                
//                 glowColor: videoPlayerController.isSstActive.value
//                     ? Color.fromRGBO(32, 193, 148, 100)
//                     : Color.fromRGBO(32, 193, 148, 100),
//                 duration: const Duration(milliseconds: 2000),
//                 // repeat: true,
//                 glowCount: 2,
//                 startDelay: const Duration(milliseconds: 100),
//                 animate: videoPlayerController.isSstActive.value,
//                 child: CircleAvatar(
//                   backgroundColor: Colors.white,
//                   radius: 50.0,
//                   child: Obx(
//                     () => Icon(Icons.mic,
//                         size: 50,
//                         color: videoPlayerController.isSstActive.value
//                             ? Color.fromRGBO(32, 193, 148, 100)
//                             : Color.fromRGBO(32, 193, 148, 100)),
//                   ),
//                 ),
//               ),
//             ),
//             SizedBox(
//               height: 20,
//             ),
//             Obx(() => Text(
//                   videoPlayerController.recognizedWords.value,
//                   style: TextStyle(
//                     fontSize: 19,
//                   ),
//                 ))
//           ]),
//         ],
//       ),
//     );
//   }
// }
