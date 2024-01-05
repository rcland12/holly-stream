<!doctype html>
<html>

    <head>
        <title>
            Tropa Peru 2023
        </title>
            <link rel="shortcut icon" type="image/x-icon" href="images/favicon.ico" />
            <meta http-equiv="content-language" content="en-us">
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width">
            <link rel="stylesheet" type="text/css" href="stylesheet.css">
            <script type="text/javascript">
                function changeContent() {
                    document.getElementById("button-buffer").innerHTML = `
                        <form action="login.php" method="post">
                            <input type="text" name="username" placeholder="Enter username...">
                            <input type="password" name="password" placeholder="Enter password...">
                            <button type="submit" name="submit_login" class="animated-button" id="login-button">
                                <span></span>
                                <span></span>
                                <span></span>
                                <span></span>
                                Login
                            </button>
                        </form>
                    `
                }
            </script>
    </head>

    <body>
        <div id="button-buffer">
            <a onClick="changeContent()" style="cursor: pointer;" class="animated-button">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <div id="login-div">
                    <div>
                        LOGIN
                    </div>
                    <img src="images/lock.png">
                </div>
            </a>
            <div id="error-message">
                <?php
                    session_start();
                    if(isset($_SESSION['error'])) {
                        echo $_SESSION['error'];
                    }
                    session_destroy();
                ?>
            </div>
        </div>
    </body>

</html>