#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "json.hpp"
#include "spline.h"

// for convenience
using json = nlohmann::json;

// For converting back && forth between radians && degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) 
{
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_first_of("}");
    if (found_null != std::string::npos) 
    {
        return "";
    } 
    else if (b1 != std::string::npos && b2 != std::string::npos) 
    {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

int ClosestWaypoint(double x, double y, const std::vector<double> &maps_x, const std::vector<double> &maps_y)
{
	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x, y, map_x, map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}
	}
	return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y - y), (map_x - x));

	double angle = fabs(theta - heading);
    angle = std::min(2*pi() - angle, angle);

    if(angle > pi() / 4)
    {
        closestWaypoint++;
        if (closestWaypoint == maps_x.size())
        {
            closestWaypoint = 0;
        }
    }
    return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
std::vector<double> getFrenet(double x, double y, double theta, const std::vector<double> &maps_x, const std::vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x, y, theta, maps_x, maps_y);

	int prev_wp;
	prev_wp = next_wp - 1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp] - maps_x[prev_wp];
	double n_y = maps_y[next_wp] - maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y);
	double proj_x = proj_norm * n_x;
	double proj_y = proj_norm * n_y;

	double frenet_d = distance(x_x, x_y, proj_x, proj_y);

	//see if d value is positive or negative by comparing it to a center point
	double center_x = 1000 - maps_x[prev_wp];
	double center_y = 2000 - maps_y[prev_wp];
	double centerToPos = distance(center_x, center_y, x_x, x_y);
	double centerToRef = distance(center_x, center_y, proj_x, proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i], maps_y[i], maps_x[i+1], maps_y[i+1]);
	}

	frenet_s += distance(0, 0, proj_x, proj_y);

	return {frenet_s, frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
std::vector<double> getXY(double s, double d, const std::vector<double> &maps_s, const std::vector<double> &maps_x, const std::vector<double> &maps_y)
{
	int prev_wp = -1;

	while (s > maps_s[prev_wp+1] && prev_wp < (int)(maps_s.size() - 1))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp + 1) % maps_x.size();

	double heading = atan2((maps_y[wp2] - maps_y[prev_wp]), (maps_x[wp2] - maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s - maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp] + seg_s * cos(heading);
	double seg_y = maps_y[prev_wp] + seg_s * sin(heading);

	double perp_heading = heading - pi()/2;

	double x = seg_x + d * cos(perp_heading);
	double y = seg_y + d * sin(perp_heading);

	return {x, y};
}


int changeLane(int current_lane, bool is_vehicle_close, bool is_left_vehicle_close, bool is_right_vehicle_close,
                                    double left_distance, double right_distance){

    int lane = current_lane;
    if (!is_vehicle_close) {
        if (!is_right_vehicle_close && lane == 0) {
            lane++;
        } else if (!is_left_vehicle_close && lane == 2){
            lane--;
        }
    } else {
        if (!is_left_vehicle_close && !is_right_vehicle_close){
            lane = (left_distance > right_distance) ? lane - 1 : lane + 1;
        }
        else if (!is_left_vehicle_close && lane != 0){
            lane--;
        }
        else if (!is_right_vehicle_close && lane != 2){
            lane++;
        }
    }
    return lane;
}

int main()
{
    uWS::Hub h;

    // Load up map values for waypoint's x,y,s && d normalized normal vectors
    std::vector<double> map_waypoints_x;
    std::vector<double> map_waypoints_y;
    std::vector<double> map_waypoints_s;
    std::vector<double> map_waypoints_dx;
    std::vector<double> map_waypoints_dy;

    // Waypoint map to read from
    std::string map_file_ = "../data/highway_map.csv";
    // The max s value before wrapping around the track back to 0
    double max_s = 6945.554;

    std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

    std::string line;
    while (getline(in_map_, line))
    {
        std::istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        map_waypoints_x.push_back(x);
        map_waypoints_y.push_back(y);
        map_waypoints_s.push_back(s);
        map_waypoints_dx.push_back(d_x);
        map_waypoints_dy.push_back(d_y);
    }

    // start
    int lane = 1;
    double desired_speed = 0;

    h.onMessage([&lane, &desired_speed, &map_waypoints_x, &map_waypoints_y, 
                 &map_waypoints_s, &map_waypoints_dx, &map_waypoints_dy]
                 (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode)
    {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        //auto sdata = string(data).substr(0, length);
        //cout << sdata << endl;
        if (length && length > 2 && data[0] == '4' && data[1] == '2')
        {
            auto s = hasData(data);

            if (s != "") 
            {
                auto j = json::parse(s);
                
                std::string event = j[0].get<std::string>();
                
                if (event == "telemetry")
                {
                    double car_x = j[1]["x"];
                    double car_y = j[1]["y"];
                    double car_s = j[1]["s"];
                    double car_d = j[1]["d"];
                    double car_yaw = j[1]["yaw"];
                    double car_speed = j[1]["speed"];
                    // Previous path data given to the Planner
                    auto previous_path_x = j[1]["previous_path_x"];
                    auto previous_path_y = j[1]["previous_path_y"];
                    // Previous path's end s && d values 
                    double end_path_s = j[1]["end_path_s"];
                    double end_path_d = j[1]["end_path_d"];
                    // Sensor Fusion Data, a list of all other cars on the same side of the road.
                    auto sensor_fusion = j[1]["sensor_fusion"];

                    int prev_size = previous_path_x.size();

                    if (prev_size > 0) {
                        car_s = end_path_s;
                    }

                    bool is_left_vehicle_close = false; 
                    bool is_right_vehicle_close = false; 
                    bool is_vehicle_close = false; 

                    double left_distance = 1000; 
                    double right_distance = 1000; 
                    double closest_vehicle_speed = 0;  

                    double speed_limit = 49.5;
                    double speed_change = 0.225;
                    
                    double time_step = 0.01;

                    double min_distance_infront = 20;
                    double min_distance_back = 10;


                    for (int i = 0; i < sensor_fusion.size(); i++)
                    {
                    
                        int vehicle_lane = 0;

                        double x_vehicle = sensor_fusion[i][3];
                        double y_vehicle = sensor_fusion[i][4];
                        
                        double s_vehicle  = sensor_fusion[i][5];
                        double d_vehicle  = sensor_fusion[i][6];

                        double vehicle_speed = sqrt(pow(x_vehicle, 2) + pow(y_vehicle, 2));

                        s_vehicle = s_vehicle + time_step * double(prev_size) * vehicle_speed;

                        double car_dist = s_vehicle - car_s;

                        if (lane == 2) {
                            is_right_vehicle_close = true;
                        }

                        if (lane == 0) {
                            is_left_vehicle_close = true;
                        }


                        if (car_dist > -min_distance_back && car_dist < min_distance_infront)
                        {
                            if(d_vehicle > 0 && d_vehicle <= 4) {
                                vehicle_lane = 0;
                            }
                            else if (d_vehicle > 4 && d_vehicle <= 8) { 
                                vehicle_lane = 1;
                            }
                            else if (d_vehicle > 8 && d_vehicle <= 12) {
                                vehicle_lane = 2;
                            } else continue;

                            if (vehicle_lane == lane - 1) {
                                is_left_vehicle_close = true;
                            }
                            if (vehicle_lane == lane + 1) {
                                is_right_vehicle_close = true;
                            }
                        }

                        if(car_dist > 0){

                            if (vehicle_lane == lane && car_dist < min_distance_infront){
                                is_vehicle_close = true;
                                closest_vehicle_speed = vehicle_speed;
                            }
                            if (vehicle_lane == lane - 1 && car_dist < left_distance){
                                left_distance = car_dist;
                            }
                            if (vehicle_lane == lane + 1 && car_dist < right_distance){
                                right_distance = car_dist;
                            }

                        }
                    }

                    lane = changeLane(lane, is_vehicle_close, is_left_vehicle_close, is_right_vehicle_close,
                                                                    left_distance, right_distance);
                    if (is_vehicle_close) {
                        desired_speed = (car_speed > closest_vehicle_speed) ? desired_speed - speed_change : desired_speed + speed_change;
                    }
                    else if (desired_speed < speed_limit) {
                        desired_speed += speed_change;
                    }





                    // generate trajectory part
                    std::vector<double> ptsx;
                    std::vector<double> ptsy;

                    double ref_x = car_x;
                    double ref_y = car_y;
                    double ref_yaw = deg2rad(car_yaw);
                    
                    if (prev_size < 2)
                    {
                        double prev_car_x = car_x - cos(car_yaw);
                        double prev_car_y = car_y - sin(car_yaw);

                        ptsx.push_back(prev_car_x);
                        ptsx.push_back(car_x);

                        ptsy.push_back(prev_car_y);
                        ptsy.push_back(car_y);
                    }
                    else
                    {
                        ref_x = previous_path_x[prev_size-1];
                        ref_y = previous_path_y[prev_size-1];

                        double ref_x_prev = previous_path_x[prev_size-2];
                        double ref_y_prev = previous_path_y[prev_size-2];
                        ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

                        ptsx.push_back(ref_x_prev);
                        ptsx.push_back(ref_x);

                        ptsy.push_back(ref_y_prev);
                        ptsy.push_back(ref_y);
                    }

                    std::vector<double> next_wp0 = getXY(car_s+30, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    std::vector<double> next_wp1 = getXY(car_s+60, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
                    std::vector<double> next_wp2 = getXY(car_s+90, (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

                    ptsx.push_back(next_wp0[0]);
                    ptsx.push_back(next_wp1[0]);
                    ptsx.push_back(next_wp2[0]);

                    ptsy.push_back(next_wp0[1]);
                    ptsy.push_back(next_wp1[1]);
                    ptsy.push_back(next_wp2[1]);

                    for (int i = 0; i < ptsx.size(); i++)
                    {
                        double shift_x = ptsx[i] - ref_x;
                        double shift_y = ptsy[i] - ref_y;

                        ptsx[i] = (shift_x * cos(-ref_yaw) - shift_y * sin(-ref_yaw));
                        ptsy[i] = (shift_x * sin(-ref_yaw) + shift_y * cos(-ref_yaw));
                    }

                    tk::spline s;
                    s.set_points(ptsx, ptsy);
                    
                    std::vector<double> next_x_vals;
                    std::vector<double> next_y_vals;
                    
                    for (int i = 0; i < previous_path_x.size(); i++)
                    {
                        next_x_vals.push_back(previous_path_x[i]);
                        next_y_vals.push_back(previous_path_y[i]);
                    }

                    double target_x = 30;
                    double target_y = s(target_x);
                    double target_dist = sqrt(target_x*target_x + target_y*target_y);

                    double x_add_on = 0;

                    for (int i = 1; i <= 50 - previous_path_x.size(); i++)
                    {
                        double N = target_dist / (0.02 * desired_speed / 2.24);
                        double x_point = x_add_on + target_x / N;
                        double y_point = s(x_point);

                        x_add_on = x_point;

                        double x_ref = x_point;
                        double y_ref = y_point;

                        x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
                        y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

                        x_point += ref_x;
                        y_point += ref_y;

                        next_x_vals.push_back(x_point);
                        next_y_vals.push_back(y_point);
                    }

                    json msgJson;
                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;

                    auto msg = "42[\"control\","+ msgJson.dump()+"]";
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                    
                }
            } 
            else
            {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });

    // We don't need ego.ince we're not using HTTP but if it's removed the
    // program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t)
    {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1)
        {
            res->end(s.data(), s.length());
        }
        else
        {
            // i guess ego.hould be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req)
    {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length)
    {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port))
    {
        std::cout << "Listening to port " << port << std::endl;
    }
    else
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
