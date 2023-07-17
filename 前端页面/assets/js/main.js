(function($) {

	"use strict";

	/*
	|--------------------------------------------------------------------------
	| Template Name: Trustlife
	| Author: ThemeMarch
	| Developer: Tamjid Bin Murtoza
	| Version: 1.0.0
	|--------------------------------------------------------------------------
	|--------------------------------------------------------------------------
	| TABLE OF CONTENTS:
	|--------------------------------------------------------------------------
	|
	| 1. Scripts initialization
	| 2. Preloader
	| 3. Primary Menu
	| 4. Scroll Function
	| 5. Section Active and Scrolling Animation
	| 6. Accordian
	| 7. Tab
	| 8. Custom Select 
	| 9. Scroll Up
	| 10. Owl Carousel
	| 11. Portfolio
	| 12. Magnific Popup
	| 13. Ajax Contact Form And Appointment
	| 14. Mailchimp js
	| 15. Google Map
	|
	*/

	/*--------------------------------------------------------------
		1. Scripts initialization
	--------------------------------------------------------------*/
	$.exists = function(selector) {
        return ($(selector).length > 0);
    }
    
	$(window).on('load', function() {
		$(window).trigger("scroll");
		$(window).trigger("resize");
		preloaderSetup();
		portfolioMsSetup();
	});

	$(document).ready(function() {
		$(window).trigger("resize");
		primaryMenuSetup();
		mobileMenu();
		scrollAnimation();
		accordianSetup();
		tabSetup();
		scrollUp();
		customSelectSetup();
		owlCarouselSetup();
		portfolioMsSetup();
		magnificPopupSetup();
		rippleSetup();
		contactForm();
		appointmentForm();
		onePageNavigation();
		new WOW().init();
		$('#udate').datepicker();
		$('.tm-counter').tamjidCounter();
		$('.tm-video-button').modalVideo();
		$(".player").YTPlayer();
		$('.parallax').parallax("50%", 0.3);
	});

	$(window).on('resize', function() {
		mobileMenu();
		portfolioMsSetup();
	});

	$(window).on('scroll', function() {
		scrollFunction();
	});


	/*--------------------------------------------------------------
		2. Preloader
	--------------------------------------------------------------*/

	function preloaderSetup() {
		$("#tm-preloader-in").fadeOut();
		$("#tm-preloader").delay(150).fadeOut("slow");
	}


	/*--------------------------------------------------------------
		3. Primary Menu
	--------------------------------------------------------------*/
	
	function primaryMenuSetup() {

		$( ".tm-primary-nav-list" ).before( "<div class='m-menu-btn'><span></span></div>" );

		$(".m-menu-btn").on('click', function(){
			$( this ).toggleClass( "m-menu-btn-ext" );
			$(this).siblings('.tm-primary-nav-list').slideToggle("slow");
		});

		$( ".menu-item-has-children > ul" ).before( "<i class='fa fa-plus m-dropdown'></i>" );

		$('.m-dropdown').on('click', function() {
			$(this).siblings('ul').slideToggle("slow");
			$(this).toggleClass("fa-plus fa-minus")
		});


		$('.maptoggle').on('click', function() {
			$( this ).siblings('.google-map').toggleClass('map-toggle');
		});

	}

	function mobileMenu() {

		if ($(window).width() <= 991){  
			$('.tm-primary-nav').addClass('m-menu').removeClass('tm-primary-nav');
		} else {
			$('.m-menu').addClass('tm-primary-nav').removeClass('m-menu');
		}

	}

	/*--------------------------------------------------------------
    One Page Navigation
  --------------------------------------------------------------*/
  function onePageNavigation() {
    // Click To Go Top
    $('.tm-smooth-move').on('click', function () {
      var thisAttr = $(this).attr('href');
      if ($(thisAttr).length) {
        var scrollPoint = $(thisAttr).offset().top - 50;
        $('body,html').animate({
          scrollTop: scrollPoint
        }, 500);
      }
      return false;
    });

    // One Page Active Class
    var topLimit = 300,
      ultimateOffset = 200;

    $('.tm-onepage-nav').each(function () {
      var $this = $(this),
        $parent = $this.parent(),
        current = null,
        $findLinks = $this.find("a");

      function getHeader(top) {
        var last = $findLinks.first();
        if (top < topLimit) {
          return last;
        }
        for (var i = 0; i < $findLinks.length; i++) {
          var $link = $findLinks.eq(i),
            href = $link.attr("href");

          if (href.charAt(0) === "#" && href.length > 1) {
            var $anchor = $(href).first();
            if ($anchor.length > 0) {
              var offset = $anchor.offset();
              if (top < offset.top - ultimateOffset) {
                return last;
              }
              last = $link;
            }
          }
        }
        return last;
      }

      $(window).on("scroll", function () {
        var top = window.scrollY,
          height = $this.outerHeight(),
          max_bottom = $parent.offset().top + $parent.outerHeight(),
          bottom = top + height + ultimateOffset;

        var $current = getHeader(top);

        if (current !== $current) {
          $this.find(".active").removeClass("active");
          $current.addClass("active");
          current = $current;
        }
      });
    });
  }

	/*--------------------------------------------------------------
		4. Scroll Function
	--------------------------------------------------------------*/

	function scrollFunction() {

		var scroll = $(window).scrollTop();

		if(scroll >= 10) {
				$(".tm-site-header").addClass("small-height");
			} else {
					$(".tm-site-header").removeClass("small-height");
			}

		// For Scroll Up
		if(scroll >= 350) {
				$("#scrollup").addClass("scrollup-show");
			} else {
					$("#scrollup").removeClass("scrollup-show");
			}

	}

	/*--------------------------------------------------------------
		5. Section Active and Scrolling Animation
	--------------------------------------------------------------*/

	function scrollAnimation() {

		$('a:not(.tm-portfolio-filter ul li a):not(.tm-portfolio-item .item-inner):not(.services a)').on('click', function(event) {
			var $anchor = $(this);
			$('html, body').stop().animate({
				scrollTop: ($($anchor.attr('href')).offset().top - 30)
				}, 1250, 'easeInOutExpo');
				event.preventDefault();
		});

	}

	/*--------------------------------------------------------------
		6. Accordian
	--------------------------------------------------------------*/

	function accordianSetup() {

        var $this = $(this);
        $( ".accordian-head" ).append( "<span class='accordian-toggle'></span>" );
        $('.single-accordian').filter(':nth-child(n+2)').children('.accordian-body').hide();
        $('.single-accordian:first-child').children('.accordian-head').addClass('active');
        $('.accordian-head').on('click', function() {
            $(this).parent('.single-accordian').siblings().children('.accordian-body').slideUp();
            $(this).siblings().slideToggle();
            /* Accordian Active Class */
            $(this).toggleClass('active');
            $(this).parent('.single-accordian').siblings().children('.accordian-head').removeClass('active');
        });

    }

    /*--------------------------------------------------------------
		7. Tab
	--------------------------------------------------------------*/
	
	function tabSetup() {
		// Custom Tab
		$('.tm-tab-wrap ul.tabs').addClass('active').find('> li:eq(0)').addClass('current');
		$('.tm-tab-wrap ul.tabs li').on('click', function (g) { 
			var tab = $(this).closest('.tm-tab-wrap'), 
				index = $(this).closest('li').index();
			
			tab.find('ul.tabs > li').removeClass('current');
			$(this).closest('li').addClass('current');
			
			tab.find('.tm-tab-content').find('div.tm-tabs-item').not('div.tm-tabs-item:eq(' + index + ')').slideUp(700);
			tab.find('.tm-tab-content').find('div.tm-tabs-item:eq(' + index + ')').slideDown(700);
			
			g.preventDefault();
		} );
	}


	/*--------------------------------------------------------------
		8. Custom Select 
	--------------------------------------------------------------*/

	function customSelectSetup() {

		$(".tm-custom-select").each(function() {
		  	var classes = $(this).attr("class"),
		     	id      = $(this).attr("id"),
		    	name    = $(this).attr("name");
		  	var template =  '<div class="' + classes + '">';
		      	template += '<span class="custom-select-trigger">' + '<div>' + $(".tm-custom-select-wrap > .tm-custom-select option:first").html() + '</div>' + '</span>';
		     	template += '<div class="custom-options">';
		    	$(this).find("option").each(function() {
		    		template += '<span class="custom-option ' + $(this).attr("class") + '" data-value="' + $(this).attr("value") + '">' + $(this).html() + '</span>';
		     	});
		 	template += '</div></div>';
		  
		 	$(this).wrap('<div class="custom-select-wrapper"></div>');
		 	$(this).hide();
		 	$(this).after(template);
		});
		$(".custom-option:first-of-type").on('hover', function() {
		  	$(this).parents(".custom-options").addClass("option-hover");
		}, function() {
		  	$(this).parents(".custom-options").removeClass("option-hover");
		});
		$(".custom-select-trigger").on("click", function(event) {
		  	$('html').one('click',function() {
		   		$(".tm-custom-select").removeClass("opened");
		 	});
		 	$(this).parents(".tm-custom-select").toggleClass("opened");
		 	event.stopPropagation();
		});
		$(".custom-option").on("click", function() {
		 	$(this).parents(".custom-select-wrapper").find("select").val($(this).data("value"));
		 	$(this).parents(".custom-options").find(".tm-custom-option").removeClass("selection");
		 	$(this).addClass("selection");
		 	$(this).parents(".custom-select").removeClass("opened");
		 	$(this).parents(".tm-custom-select").find(".custom-select-trigger").text($(this).text());
		});

	}

	/*--------------------------------------------------------------
		9. Scroll Up
	--------------------------------------------------------------*/

	function scrollUp() {

		$('#scrollup').on('click', function(e) {
			e.preventDefault();
			$('html,body').animate({
				scrollTop: 0
			}, 1000);
		});

	}

	/*--------------------------------------------------------------
		10. Owl Carousel
	--------------------------------------------------------------*/

	function owlCarouselSetup() {

		/* Owl Carousel For tm-hero-slider */
	    $('#tm-hero-slider').owlCarousel({
	        items: 1,
            animateOut: 'fadeOut',
            animateIn: 'fadeIn',
            loop: true,
            nav: false,
            dots: true,
            autoplay: true      
        });
        /* Owl Carousel For tm-hero-slider */
	    $('#tm-hero-doctor').owlCarousel({
	        items: 1,
            animateOut: 'fadeOut',
            animateIn: 'fadeIn',
            loop: true,
            nav: false,
            dots: false,
            autoplay: true      
        });

        /* Owl Carousel For tm-hero-slider */
	    $('#hero-slider1').owlCarousel({
	        items: 1,
            loop: true,
            nav: false,
            nav:true,
            navText:['<i class="icofont icofont-bubble-left">', '<i class="icofont icofont-bubble-right">'],
            dots: false,
            smartSpeed:900,
            autoplayTimeout:3000,
            autoplay: false      
        });

        /* Owl Carousel For Services */
		$('.service-curosor').owlCarousel({
            loop:true,
            margin:20,
            nav:true,
            navText:['', ''],
            autoplay:false,
            smartSpeed:600,
            autoplayTimeout:2500,
            responsive:{
                0:{
                    items:1
                },
                575:{
                    items:2
                },
                991:{
                    items:3
                }
            }
        });

        /* Owl Carousel Team Member */
		$('.tm-member-carousel').owlCarousel({
            loop:true,
            margin:30,
            nav:true,
            navText:['<i class="icofont icofont-bubble-left"></i>', '<i class="icofont icofont-bubble-right"></i>'],
            autoplay:false,
            smartSpeed:600,
            dots: true,
            autoplayTimeout:2500,
            responsive:{
                0:{
                    items:1
                },
                500:{
                    items:2
                },
                767:{
                    items:3
                },
                1199:{
                    items:4
                }
            }
        });

        /* Owl Carousel For Client Curosor */
		$('.tm-clients-curosor').owlCarousel({
            loop:true,
            margin:30,
            nav:true,
            navText:['', ''],
            autoplay:true,
            smartSpeed:600,
            autoplayTimeout:2500,
            responsive:{
                0:{
                    items:1
                },
                500:{
                    items:2
                },
                767:{
                    items:3
                },
                991:{
                    items:4
                },
                1199:{
                    items:5
                }
            }
        });

        /* Owl Carousel For Pricing Table */
		$('.tm-pricing-carousel').owlCarousel({
            loop:true,
            margin:30,
            nav:true,
            navText:['<i class="icofont icofont-bubble-left"></i>', '<i class="icofont icofont-bubble-right"></i>'],
            autoplay:false,
            smartSpeed:600,
            dots: true,
            autoplayTimeout:2500,
            responsive:{
                0:{
                    items:1
                },
                767:{
                    items:2
                },
                991:{
                    items:3
                }
            }
        });

        /* Owl Carousel For Testimonial */
        $('.tm-testimonial-1').owlCarousel({
        	items: 1,
            animateOut: 'bounceOutRight',
            loop: true,
            autoplay: true,
            autoplayTimeout: 6000,
            smartSpeed: 1000,
            nav: false,
            dots: true    
        });

        /* Owl Carousel For Testimonial-2 */
        $('.tm-testimonial-2').owlCarousel({
        	items: 1,
            animateOut: 'slideOutDown',
            animateIn: 'flipInX',
            loop: true,
            autoplay: true,
            autoplayTimeout: 6000,
            smartSpeed: 1000,
            nav: false 
        });

        /* Owl Carousel For Blog Post Carousel */
	    $('.tm-post-carousel').owlCarousel({
	        items: 1,
            loop: true,
            nav: true,
            navText: ['<span class="icofont icofont-long-arrow-left"></span>','<span class="icofont icofont-long-arrow-right"></span>'],
            autoplay: false,
            autoplayTimeout: 6000,
            smartSpeed: 1000,
            dots: false     
		});
		
		/* Owl Carousel For Before & After */
		$('.before-after-gallery-slider').owlCarousel({
			items: 1,
			loop: false,
			nav: true,
			navText: ['<span class="icofont icofont-long-arrow-left"></span>','<span class="icofont icofont-long-arrow-right"></span>'],
			autoplay: true,
			autoplayTimeout: 6000,
			smartSpeed: 1000,
			dots: false,
			mouseDrag: false,
			touchDrag: false 
		});

	}

	/*--------------------------------------------------------------
		11. Portfolio
	--------------------------------------------------------------*/

	function portfolioMsSetup() {

		$('.tm-portfolio').isotope({
			itemSelector: '.tm-portfolio-item',
			transitionDuration: '0.60s',
			percentPosition: true,
			masonry: {
				columnWidth: '.tm-grid-sizer'
			}
		});

		/* Active Class of Portfolio*/
		$('.tm-portfolio-filter ul li').on('click', function(event) {
			$(this).siblings('.active').removeClass('active');
			$(this).addClass('active');
				event.preventDefault();
		});

		/*=== Portfolio filtering ===*/
		$('.tm-portfolio-filter ul').on('click', 'a', function() {
			var filterElement = $(this).attr('data-filter');
			$(this).parents(".tm-portfolio-filter").next().isotope({
				filter: filterElement
			});
		});

	}


  	/*--------------------------------------------------------------
   		12. Magnific Popup
  	--------------------------------------------------------------*/

  	function magnificPopupSetup() {

	    $('.zoom-gallery').magnificPopup({
	        delegate: '.item-inner',
	        type: 'image',
	        closeOnContentClick: false,
	        closeBtnInside: false,
	        mainClass: 'mfp-with-zoom mfp-img-mobile',
	        gallery: {
	            enabled: true
	        },
	        zoom: {
	            enabled: true,
	            duration: 300, // don't foget to change the duration also in CSS
	            opener: function(element) {
	                return element.find('img');
	            }
	        }
	        
	    });
	    
  	}

   	/*--------------------------------------------------------------
   		13. Magnific Popup
  	--------------------------------------------------------------*/

  	function rippleSetup() {
		
        if ($.exists('.ripple-version')) {

            $('.ripple-version').ripples({
				resolution: 500,
		        perturbance: 0.04
			});

        }
	
	}

	/*--------------------------------------------------------------
		14. Ajax Contact Form And Appointment
	--------------------------------------------------------------*/
	// Contact Form
	function contactForm() {

		$('#tm-alert').hide();
	    $('#contact-form #submit').on('click', function() {
	        var name = $('#name').val();
	        var subject = $('#subject').val();
	        var phone = $('#phone').val();
	        var email = $('#email').val();
	        var msg = $('#msg').val();
			var regex = /^([a-zA-Z0-9_.+-])+\@(([a-zA-Z0-9-])+\.)+([a-zA-Z0-9]{2,4})+$/;
			
			if (!regex.test(email)) {
				$('#tm-alert').fadeIn().html('<div class="alert alert-danger"><strong>Warning!</strong> Please Enter Valid Email.</div>');
				return false;
			}

	        name = $.trim(name);
	        subject = $.trim(subject);
	        phone = $.trim(phone);
	        email = $.trim(email);
	        msg = $.trim(msg);

	        if (name != '' && email != '' && msg != '') {
	            var values = 	"name=" + name + 
	            				"&subject=" + subject + 
	            				"&phone=" + phone + 
	            				"&email=" + email + 
	            				"&msg=" + msg;
	            $.ajax({
	                type: "POST",
	                url: "assets/php/mail.php",
	                data: values,
	                success: function() {
	                    $('#name').val('');
	                    $('#subject').val('');
	                    $('#phone').val('');
	                    $('#email').val('');
	                    $('#msg').val('');

	                    $('#tm-alert').fadeIn().html('<div class="alert alert-success"><strong>Success!</strong> Email has been sent successfully.</div>');
	                    setTimeout(function() {
	                        $('#tm-alert').fadeOut('slow');
	                    }, 4000);
	                }
	            });
	        } else {
				$('#tm-alert').fadeIn().html('<div class="alert alert-danger"><strong>Warning!</strong> All fields are required.</div>');
	        }
	        return false;
	    });

	}

	// Appointment Form
	function appointmentForm() {

		$('#tm-alert1').hide();
	    $('#appointment-form #appointment-submit').on('click', function() {
	        var uname = $('#uname').val();
	        var uemail = $('#uemail').val();
	        var unumber = $('#unumber').val();
	        var udate = $('#udate').val();
	        var udepartment = $('#udepartment').val();
	        var udoctor = $('#udoctor').val();
	        var umsg = $('#umsg').val();
			var regex = /^([a-zA-Z0-9_.+-])+\@(([a-zA-Z0-9-])+\.)+([a-zA-Z0-9]{2,4})+$/;
			
			if (!regex.test(uemail)) {
				$('#tm-alert1').fadeIn().html('<div class="alert alert-danger"><strong>Warning!</strong> Please Enter Valid Email.</div>');
				return false;
			}

	        uname = $.trim(uname);
	        uemail = $.trim(uemail);
	        unumber = $.trim(unumber);
	        udate = $.trim(udate);
	        udepartment = $.trim(udepartment);
	        udoctor = $.trim(udoctor);
	        umsg = $.trim(umsg);

	        if (uname != '' && uemail != '' && umsg != '') {
	            var values = 	"uname=" + uname + 
	            				"&uemail=" + uemail + 
	            				"&unumber=" + unumber + 
	            				"&udate=" + udate + 
	            				"&udepartment=" + udepartment + 
	            				"&udoctor=" + udoctor + 
	            				"&umsg=" + umsg;
	            $.ajax({
	                type: "POST",
	                url: "assets/php/appointment.php",
	                data: values,
	                success: function() {
	                    $('#uname').val('');
	                    $('#uemail').val('');
	                    $('#unumber').val('');
	                    $('#udepartment').val('');
	                    $('#udoctor').val('');
	                    $('#umsg').val('');

	                    $('#tm-alert1').fadeIn().html('<div class="alert alert-success"><strong>Success!</strong> Appointment has been sent successfully.</div>');
	                    setTimeout(function() {
	                        $('#tm-alert1').fadeOut('slow');
	                    }, 4000);
	                }
	            });
	        } else {
				$('#tm-alert1').fadeIn().html('<div class="alert alert-danger"><strong>Warning!</strong> All fields are required.</div>');
	        }
	        return false;
	    });

	}


	/*--------------------------------------------------------------
	    15. Mailchimp js
	--------------------------------------------------------------*/
	// mailchimp start
    if ($('.mailchimp').length > 0) {
        $('.mailchimp').ajaxChimp({
            language: 'es',
            callback: mailchimpCallback
        });
    }

    function mailchimpCallback(resp) {
        if (resp.result === 'success') {
            $('.subscription-success').html('<i class="fa fa-check"></i><br/>' + resp.msg).fadeIn(1000);
            $('.subscription-error').fadeOut(500);

        } else if (resp.result === 'error') {
            $('.subscription-error').html('<i class="fa fa-times"></i><br/>' + resp.msg).fadeIn(1000);
        }
    }
    $.ajaxChimp.translations.es = {
        'submit': 'Submitting...',
        0: 'We have sent you a confirmation email',
        1: 'Please enter a value',
        2: 'An email address must contain a single @',
        3: 'The domain portion of the email address is invalid (the portion after the @: )',
        4: 'The username portion of the email address is invalid (the portion before the @: )',
        5: 'This email address looks fake or invalid. Please enter a real email address'
    };

	/*--------------------------------------------------------------
	    17. Parallax 2
	--------------------------------------------------------------*/
	if ($.exists('#tm-scene')) {

    var scene = document.getElementById('tm-scene');
		var parallax = new Parallax(scene);

	}
	

	/*--------------------------------------------------------------
	    18. Tweenty Tweenty
	--------------------------------------------------------------*/
	$('.teeth-beforeafter').twentytwenty({
		before_label: '',
		after_label: '',
		no_overlay: true
	});



})(jQuery); // End of use strict
