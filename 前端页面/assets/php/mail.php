<?php

$to 		= 'tamjid.storerepublic@gmail.com';
$headers	= 'FROM: "'.$email.'"';

//All form values
$name 		= 	$_POST['name'];
$email 		= 	$_POST['email'];
$subject 	= 	$_POST['subject'];
$phone 		= 	$_POST['phone'];
$msg 		= 	$_POST['msg'];
$output 	= 	"Name: ".$name.
				"\nEmail: ".$email.
				"\nPhone: ".$phone.
				"\nSubject: ".$subject.
				"\n\nMessage: ".$msg;

$send		= mail($to, $name, $output, $headers);