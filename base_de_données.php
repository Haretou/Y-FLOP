-- Création des tables
CREATE TABLE Place (
    id TEXT PRIMARY KEY,
    Name VARCHAR(255),
    Latitude FLOAT,
    Longitude FLOAT,
    time_id TEXT
);

CREATE TABLE Weather (
    id TEXT PRIMARY KEY,
    State VARCHAR(255),
    Town VARCHAR(255),
    temperature FLOAT,
    precipitation FLOAT,
    wind FLOAT,
    humidity FLOAT
);

CREATE TABLE Time (
    id TEXT PRIMARY KEY,
    day TIMESTAMP,
    hour TIMESTAMP,
    weather_id TEXT,
    place_id TEXT,
    FOREIGN KEY (weather_id) REFERENCES Weather(id),
    FOREIGN KEY (place_id) REFERENCES Place(id)
);

-- Insertion d'exemples de données
INSERT INTO Place (id, Name, Latitude, Longitude, time_id) 
VALUES ('1', 'Montpellier', 43.6117, 3.8767, '1');

INSERT INTO Weather (id, State, Town, temperature, precipitation, wind, humidity)
VALUES ('1', 'Occitanie', 'Montpellier', 22.5, 0.0, 12.3, 60.0);

INSERT INTO Time (id, day, hour, weather_id, place_id) 
VALUES ('1', '2024-12-18', '10:00:00', '1', '1');

-- Requête pour afficher toutes les informations liées à une place et à sa météo
SELECT 
    Place.Name AS Place_Name, 
    Place.Latitude, 
    Place.Longitude, 
    Weather.State, 
    Weather.Town, 
    Weather.temperature, 
    Weather.precipitation, 
    Weather.wind, 
    Weather.humidity, 
    Time.day, 
    Time.hour
FROM Place
JOIN Time ON Place.id = Time.place_id
JOIN Weather ON Time.weather_id = Weather.id;

-- Requête pour trouver la météo d'une ville spécifique
SELECT 
    Weather.Town, 
    Weather.temperature, 
    Weather.wind, 
    Weather.humidity
FROM Weather
WHERE Weather.Town = 'Montpellier';
