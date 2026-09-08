#pragma once

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace dzeta {

class RangeThreadPool {
public:
    explicit RangeThreadPool(std::size_t max_workers)
        : max_workers_(std::max<std::size_t>(1, max_workers)) {
        threads_.reserve(max_workers_ > 0 ? max_workers_ - 1U : 0U);
        for (std::size_t i = 0; i + 1U < max_workers_; ++i) {
            threads_.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    RangeThreadPool(const RangeThreadPool&) = delete;
    RangeThreadPool& operator=(const RangeThreadPool&) = delete;

    ~RangeThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::size_t max_workers() const noexcept { return max_workers_; }

    template <typename Fn>
    void run(std::size_t work_items, std::size_t workers, Fn&& fn) {
        workers = std::min({workers, max_workers_, work_items});
        if (workers <= 1) {
            fn(0, work_items);
            return;
        }

        std::vector<std::pair<std::size_t, std::size_t>> local_ranges;
        local_ranges.reserve(workers);
        const std::size_t block = (work_items + workers - 1U) / workers;
        std::size_t begin = 0;
        while (begin < work_items && local_ranges.size() < workers) {
            const std::size_t end = std::min(work_items, begin + block);
            local_ranges.emplace_back(begin, end);
            begin = end;
        }
        workers = local_ranges.size();
        if (workers <= 1) {
            fn(0, work_items);
            return;
        }

        using TaskFn = std::decay_t<Fn>;
        auto task_holder = std::make_shared<TaskFn>(std::forward<Fn>(fn));
        std::pair<std::size_t, std::size_t> main_range{0, 0};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ranges_ = std::move(local_ranges);
            task_ = [task_holder](std::size_t range_begin, std::size_t range_end) {
                (*task_holder)(range_begin, range_end);
            };
            active_workers_ = workers - 1U;
            remaining_workers_ = active_workers_;
            exception_ = nullptr;
            main_range = ranges_[workers - 1U];
            ++generation_;
        }
        work_cv_.notify_all();

        std::exception_ptr main_exception;
        try {
            (*task_holder)(main_range.first, main_range.second);
        } catch (...) {
            main_exception = std::current_exception();
            std::lock_guard<std::mutex> lock(mutex_);
            if (!exception_) exception_ = main_exception;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
        auto ex = exception_;
        task_ = {};
        active_workers_ = 0;
        lock.unlock();
        if (ex) std::rethrow_exception(ex);
    }

private:
    void worker_loop(std::size_t worker_index) {
        std::size_t seen_generation = 0;
        while (true) {
            std::function<void(std::size_t, std::size_t)> task;
            std::pair<std::size_t, std::size_t> range{0, 0};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait(lock, [&]() {
                    return stop_ || generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                seen_generation = generation_;
                if (worker_index >= active_workers_) {
                    continue;
                }
                range = ranges_[worker_index];
                task = task_;
            }

            try {
                task(range.first, range.second);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!exception_) exception_ = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --remaining_workers_;
                if (remaining_workers_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::size_t max_workers_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::vector<std::pair<std::size_t, std::size_t>> ranges_;
    std::function<void(std::size_t, std::size_t)> task_;
    std::size_t active_workers_ = 0;
    std::size_t remaining_workers_ = 0;
    std::size_t generation_ = 0;
    bool stop_ = false;
    std::exception_ptr exception_;
};

} // namespace dzeta
