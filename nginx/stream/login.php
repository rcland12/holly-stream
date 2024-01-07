<?php
session_start();

require_once 'class-db.php';

$error_message = '';
if (isset($_POST['submit_login'])) {
    $db = new DB();
    $response = $db->check_credentials($_POST['username'], $_POST['password']);

    if ($response['status'] == 'success') {
        $_SESSION['login'] = true;
        header('Location: clips/index.php');
    } elseif ($response['status'] == 'error') {
        $_SESSION['login'] = false;
        $_SESSION['error'] = $response['message'];
        header('Location: index.php');
    } else {
        $_SESSION['login'] = false;
        echo 'Something went wrong, try again.';
    }
}

?>