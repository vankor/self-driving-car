#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

#define THRESHOLD 0.00001

using namespace std;

static default_random_engine random_generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {


  normal_distribution<double> theta_norm_init(0, std[2]);
  normal_distribution<double> x_norm_init(0, std[0]);
  normal_distribution<double> y_norm_init(0, std[1]);

  num_particles = 100;

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    
    particle.weight = 1.0;
    particle.id = i;
    
    particle.x = x;
    particle.y = y;
    particle.theta = theta;

    particle.theta += theta_norm_init(random_generator);
    particle.x += x_norm_init(random_generator);
    particle.y += y_norm_init(random_generator);

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  normal_distribution<double> norm_x(0, std_pos[0]);
  normal_distribution<double> norm_y(0, std_pos[1]);
  normal_distribution<double> norm_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    if (fabs(yaw_rate) < THRESHOLD) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += norm_x(random_generator);
    particles[i].y += norm_y(random_generator);
    particles[i].theta += norm_theta(random_generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs o = observations[i];

    double min_distance = numeric_limits<double>::max();

    int mapped_observation = -1;
    
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs p = predicted[j];
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      if (cur_dist < min_distance) {
        min_distance = cur_dist;
        mapped_observation = p.id;
      }
    }

    observations[i].id = mapped_observation;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  for (int i = 0; i < num_particles; i++) {

    double prediction_x = particles[i].x;
    double prediction_y = particles[i].y;
    double prediction_theta = particles[i].theta;
    
    particles[i].weight = 1.0;
    
    vector<LandmarkObs> transformed_coords;
    for (int j = 0; j < observations.size(); j++) {
      double t_x = cos(prediction_theta)*observations[j].x - sin(prediction_theta)*observations[j].y + prediction_x;
      double t_y = sin(prediction_theta)*observations[j].x + cos(prediction_theta)*observations[j].y + prediction_y;
      transformed_coords.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    vector<LandmarkObs> predictions;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      if (fabs(landmark_x - prediction_x) <= sensor_range && fabs(landmark_y - prediction_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{ lm_id, landmark_x, landmark_y });
      }
    }    

    dataAssociation(predictions, transformed_coords);


    for (int j = 0; j < transformed_coords.size(); j++) {
      
      double observation_x = transformed_coords[j].x;
      double observation_y = transformed_coords[j].y;

      double prediction_x, prediction_y;

      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];

      int nearest_pred = transformed_coords[j].id;

      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == nearest_pred) {
          prediction_x = predictions[k].x;
          prediction_y = predictions[k].y;
        }
      }

      double obs_w = ( 1/(2*M_PI*sigma_x*sigma_y)) * exp( -( pow(prediction_x-observation_x,2)/(2*pow(sigma_x, 2)) + (pow(prediction_y-observation_y,2)/(2*pow(sigma_y, 2))) ) );

      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  
  vector<Particle> new_particles;
  
  double beta = 0.0;

  vector<double> weights;
  
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  double weight_max = *max_element(weights.begin(), weights.end());

  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  int index = uniintdist(random_generator);

  uniform_real_distribution<double> unirealdist(0.0, weight_max);

  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(random_generator) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1); 
    return s;
}