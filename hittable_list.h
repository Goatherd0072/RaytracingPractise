#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include"hittable.h"

#include<memory>
#include<vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable
{
public:
	hittable_list() {}
	hittable_list(shared_ptr<hittable > object) { add(object); }

	void clear() { objects.clear(); }
	void add(shared_ptr<hittable> object) { objects.push_back(object); }//把object压入vector
	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;

public:
	std::vector<shared_ptr<hittable>> objects;
};
bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& record) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects) //遍历object，寻找最接近的t_max(c++11 特性，基于范围的for循环)
    {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) 
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            record = temp_rec;
        }
    }
    return hit_anything;
}


#endif // !HITTABLE_LIST_H
