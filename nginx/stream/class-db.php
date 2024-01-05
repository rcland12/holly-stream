<?php
class DB {
    private $dbHost     = "perugoproclips.com";
    private $dbUsername = "roooooooooot";
    private $dbPassword = "Gu1neaB1ssau6yhn^YHN7ujm&UJM";
    private $dbName     = "user_login";

    public function __construct() {
        if(!isset($this->db)){
            $conn = new mysqli($this->dbHost, $this->dbUsername, $this->dbPassword, $this->dbName);
            if($conn->connect_error){
                die("Failed to connect with MySQL: " . $conn->connect_error);
            }else{
                $this->db = $conn;
            }
        }
    }

    public function check_credentials($username = '', $password = '') {
        $sql = $this->db->query("SELECT id FROM users WHERE username = '$username' AND password = '". md5($password) ."'");

        if($sql->num_rows) {
            $result = $sql->fetch_assoc();
            return array('status' => 'success', 'username' => $result['username']);
        }

        return array('status' => 'error', 'message' => 'Invalid username or password.');
    }
}
?>