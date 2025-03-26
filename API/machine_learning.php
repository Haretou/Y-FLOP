<?php
require_once __DIR__ . '/../config.php';

/**
 * Fonctions pour l'apprentissage automatique et l'analyse des données météorologiques
 */

/**
 * Obtient les tendances de température pour une ville sur les dernières recherches
 * 
 * @param PDO $pdo Instance de connexion PDO
 * @param string $city Nom de la ville
 * @return array Données de tendance
 */
function getTemperatureTrend($pdo, $city) {
    if (!$pdo) {
        return ['error' => 'Connexion à la base de données impossible'];
    }
    
    try {
        // Récupérer l'historique des températures pour cette ville
        $stmt = $pdo->prepare(
            "SELECT temperature, search_date 
             FROM searches 
             WHERE city = :city 
             ORDER BY search_date DESC 
             LIMIT 30"
        );
        $stmt->execute(['city' => $city]);
        $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        // Calculer les tendances
        if (count($results) > 1) {
            $firstTemp = $results[count($results) - 1]['temperature'];
            $lastTemp = $results[0]['temperature'];
            $trend = $lastTemp - $firstTemp;
            
            return [
                'city' => $city,
                'current' => $lastTemp,
                'trend' => $trend,
                'trend_direction' => ($trend > 0) ? 'hausse' : (($trend < 0) ? 'baisse' : 'stable'),
                'data_points' => count($results),
                'history' => array_reverse($results)
            ];
        } else {
            return [
                'city' => $city,
                'error' => 'Pas assez de données pour calculer une tendance'
            ];
        }
    } catch (PDOException $e) {
        return ['error' => 'Erreur lors de la récupération des tendances: ' . $e->getMessage()];
    }
}

/**
 * Prédit la météo pour les prochains jours en se basant sur l'historique
 * 
 * @param PDO $pdo Instance de connexion PDO
 * @param string $city Nom de la ville
 * @return array Prédiction météo
 */
function predictWeather($pdo, $city) {
    if (!$pdo) {
        return ['error' => 'Connexion à la base de données impossible'];
    }
    
    try {
        // Obtenir les moyennes de température, précipitations, etc. pour cette ville par jour
        $stmt = $pdo->prepare(
            "SELECT 
                AVG(temperature) as avg_temp, 
                AVG(precipitation) as avg_precip,
                AVG(humidity) as avg_humidity,
                AVG(wind) as avg_wind,
                DATE_FORMAT(forecast_date, '%m-%d') as month_day
             FROM weather_forecasts
             WHERE city = :city 
             GROUP BY month_day"
        );
        $stmt->execute(['city' => $city]);
        $averages = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        // Obtenir les prévisions actuelles
        $stmt = $pdo->prepare(
            "SELECT * FROM weather_forecasts 
             WHERE city = :city 
             AND forecast_date >= CURDATE() 
             ORDER BY forecast_date 
             LIMIT 7"
        );
        $stmt->execute(['city' => $city]);
        $currentForecasts = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        // Ajuster les prévisions actuelles en se basant sur les tendances historiques
        $adjustedForecasts = [];
        
        foreach ($currentForecasts as $forecast) {
            $monthDay = date('m-d', strtotime($forecast['forecast_date']));
            
            // Chercher des données historiques pour ce jour du mois
            $historicalData = null;
            foreach ($averages as $avg) {
                if ($avg['month_day'] === $monthDay) {
                    $historicalData = $avg;
                    break;
                }
            }
            
            // Ajuster la prévision si des données historiques existent
            if ($historicalData) {
                $tempAdjustment = ($historicalData['avg_temp'] * 0.3 + $forecast['temperature'] * 0.7);
                $forecast['adjusted_temperature'] = round($tempAdjustment, 1);
                $forecast['confidence'] = 'moyenne';
                $forecast['historical_data'] = $historicalData;
            } else {
                $forecast['adjusted_temperature'] = $forecast['temperature'];
                $forecast['confidence'] = 'faible';
            }
            
            $adjustedForecasts[] = $forecast;
        }
        
        return [
            'city' => $city,
            'adjusted_forecasts' => $adjustedForecasts,
            'data_quality' => (count($averages) > 10) ? 'bonne' : 'limitée'
        ];
    } catch (PDOException $e) {
        return ['error' => 'Erreur lors de la prédiction météo: ' . $e->getMessage()];
    }
}

/**
 * Identifie les anomalies météorologiques (valeurs inhabituelles) pour une ville
 * 
 * @param PDO $pdo Instance de connexion PDO
 * @param string $city Nom de la ville
 * @return array Liste d'anomalies détectées
 */
function detectWeatherAnomalies($pdo, $city) {
    if (!$pdo) {
        return ['error' => 'Connexion à la base de données impossible'];
    }
    
    try {
        // Obtenir les statistiques de température pour cette ville
        $stmt = $pdo->prepare(
            "SELECT 
                AVG(temperature) as avg_temp,
                STDDEV(temperature) as std_temp,
                MAX(temperature) as max_temp,
                MIN(temperature) as min_temp
             FROM searches
             WHERE city = :city"
        );
        $stmt->execute(['city' => $city]);
        $stats = $stmt->fetch(PDO::FETCH_ASSOC);
        
        if (!$stats || $stats['avg_temp'] === null) {
            return [
                'city' => $city,
                'error' => 'Pas assez de données pour détecter des anomalies'
            ];
        }
        
        // Obtenir les dernières données météo
        $stmt = $pdo->prepare(
            "SELECT * FROM searches 
            WHERE city = :city 
            ORDER BY search_date DESC 
            LIMIT 10"
        );
        $stmt->execute(['city' => $city]);
        $recentData = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        // Détecter les anomalies (valeurs à plus de 2 écarts-types de la moyenne)
        $anomalies = [];
        foreach ($recentData as $data) {
            $zScore = ($data['temperature'] - $stats['avg_temp']) / ($stats['std_temp'] ?: 1);
            
            if (abs($zScore) > 2) {
                $anomalies[] = [
                    'date' => $data['search_date'],
                    'temperature' => $data['temperature'],
                    'z_score' => $zScore,
                    'severity' => abs($zScore) > 3 ? 'élevée' : 'modérée'
                ];
            }
        }
        
        return [
            'city' => $city,
            'anomalies' => $anomalies,
            'stats' => $stats,
            'anomaly_count' => count($anomalies)
        ];
    } catch (PDOException $e) {
        return ['error' => 'Erreur lors de la détection d\'anomalies: ' . $e->getMessage()];
    }
}
?>