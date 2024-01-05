<?php 
    session_start();
    if ($_SESSION['login']) {
?>

<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8">
		<title>
			Live Stream
		</title>
		<link href="http://localhost/js/video-js.css" rel="stylesheet" />
	</head>

	<body style="display: flex; justify-content: center; align-items: center;">
		<video
			id="my-video"
			class="video-js"
			controls
			preload="auto"
			width="1280"
			height="720"
			data-setup="{}">
			<source src="http://localhost/hls/stream.m3u8" type="application/vnd.apple.mpegurl" />
		</video>
		<script src="http://localhost/js/video.js"></script>
	</body>
</html>

<?php
}else{
    header("Location: ../index.php");
    exit();
}
?>