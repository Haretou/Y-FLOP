<?php

// Global configuration for Weather API project
return [
    'api_base_url' => 'https://api.weatherapi.com/v1/',
    'api_key' => 'your_api_key_here',
    'default_location' => 'Paris, FR',
    'units' => 'metric',
    'cache_enabled' => true,
    'cache_duration' => 3600,
    'error_handling' => [
        'log_errors' => true,
        'log_file' => __DIR__ . '/logs/error.log',
    ],
    'timeout' => 10,
    'debug' => false,
];

?>