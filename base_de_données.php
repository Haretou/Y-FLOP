CREATE TABLE Place (
    id VARCHAR(255) PRIMARY KEY,
    Name VARCHAR(255),
    Latitude FLOAT,
    Longitude FLOAT,
    time_id VARCHAR(255)
);

CREATE TABLE Weather (
    id VARCHAR(255) PRIMARY KEY,
    State VARCHAR(255),
    Town VARCHAR(255),
    temperature FLOAT,
    precipitation FLOAT,
    wind FLOAT,
    humidity FLOAT
);

CREATE TABLE Time (
    id INT PRIMARY KEY AUTO_INCREMENT,
    day DATE,
    hour TIME,
    weather_id VARCHAR(255),
    place_id VARCHAR(255),
    FOREIGN KEY (weather_id) REFERENCES Weather(id),
    FOREIGN KEY (place_id) REFERENCES Place(id)
);

-- Insertion de données dans les tables
INSERT INTO Place (id, Name, Latitude, Longitude, time_id) 
VALUES ('1', 'Montpellier', 43.6117, 3.8767, '1');

INSERT INTO Weather (id, State, Town, temperature, precipitation, wind, humidity)
VALUES ('1', 'Occitanie', 'Montpellier', 22.5, 0.0, 12.3, 60.0);

INSERT INTO Time (id, day, hour, weather_id, place_id) 
VALUES (1, '2024-12-18', '10:00:00', '1', '1');

-- Requête pour afficher toutes les informations
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