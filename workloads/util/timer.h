
#ifndef _TIMER_H_
#define _TIMER_H_ 1

#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>

/* High-resolution timer (CLOCK_MONOTONIC)
 * Uses clock_gettime instead of gettimeofday for:
 *   - nanosecond resolution (vs microsecond)
 *   - monotonic clock (immune to NTP adjustments)
 *   - lower overhead (~20ns vs ~200ns per call)
 */

typedef struct Timer {
    struct timespec startTime;
    struct timespec endTime;
} Timer;

static void startTimer(Timer* timer) {
    clock_gettime(CLOCK_MONOTONIC, &(timer->startTime));
}

static void stopTimer(Timer* timer) {
    clock_gettime(CLOCK_MONOTONIC, &(timer->endTime));
}

static double getElapsedTime(Timer timer) {
    return (double)(timer.endTime.tv_sec - timer.startTime.tv_sec)
         + (double)(timer.endTime.tv_nsec - timer.startTime.tv_nsec) / 1.0e9;
}

static void printElapsedTime(Timer timer) {
    printf("Elapsed time: %f ms\n", getElapsedTime(timer) * 1e3);
}

/* Accumulate elapsed time into a global (used by backprop per-layer timing) */
static double _accumulated_time_s = 0.0;

static void incrementElapsedTime(Timer timer) {
    _accumulated_time_s += getElapsedTime(timer);
}

static double getAccumulatedTime(void) {
    return _accumulated_time_s;
}

/* Inline RAPL energy measurement
 * Reads Intel RAPL counters via the powercap sysfs interface.
 * Uses a background polling thread to handle counter wraparound
 * on long-running workloads (counter wraps every ~65 kJ).
 *
 * Usage:
 *   EnergyCounter ec;
 *   startEnergy(&ec);
 *   ... kernel ...
 *   stopEnergy(&ec);
 *   printf("Energy: %f J\n", getElapsedEnergy(ec));
 */

static int64_t _readRaplEnergy_uj(void) {
    int64_t uj = 0;
    FILE *f = fopen("/sys/class/powercap/intel-rapl:0/energy_uj", "r");
    if (f) {
        if (fscanf(f, "%ld", &uj) != 1) uj = 0;
        fclose(f);
    }
    return uj;
}

static int64_t _readRaplMaxEnergy_uj(void) {
    int64_t max_uj = 0;
    FILE *f = fopen("/sys/class/powercap/intel-rapl:0/max_energy_range_uj", "r");
    if (f) {
        if (fscanf(f, "%ld", &max_uj) != 1) max_uj = 0;
        fclose(f);
    }
    return max_uj;
}

/* RAPL polling thread
 * Polls the RAPL counter every POLL_INTERVAL seconds and accumulates
 * energy, correctly handling wraparound across polls.
 */

#define RAPL_POLL_INTERVAL_S 10

typedef struct EnergyCounter {
    /* Polling thread state */
    pthread_t       poll_thread;
    volatile int    running;        /* flag to stop the thread       */
    int64_t         max_energy_uj;  /* counter wraparound threshold  */
    /* Accumulated result (written by poll thread, read after join) */
    volatile double total_energy_j;
} EnergyCounter;

static void* _rapl_poll_fn(void* arg) {
    EnergyCounter* ec = (EnergyCounter*)arg;
    int64_t prev = _readRaplEnergy_uj();
    const struct timespec tick = { 0, 100000000 }; /* 100ms tick */
    int ticks = 0;
    const int ticks_per_poll = RAPL_POLL_INTERVAL_S * 10;

    while (ec->running) {
        nanosleep(&tick, NULL);
        if (!ec->running) break;
        ticks++;
        if (ticks < ticks_per_poll) continue;
        ticks = 0;

        int64_t cur = _readRaplEnergy_uj();
        int64_t diff = cur - prev;
        if (diff < 0) diff += ec->max_energy_uj;
        ec->total_energy_j += (double)diff / 1.0e6;
        prev = cur;
    }

    /* Final read after stop signal */
    int64_t cur = _readRaplEnergy_uj();
    int64_t diff = cur - prev;
    if (diff < 0) diff += ec->max_energy_uj;
    ec->total_energy_j += (double)diff / 1.0e6;

    return NULL;
}

static void startEnergy(EnergyCounter* ec) {
    ec->total_energy_j = 0.0;
    ec->max_energy_uj = _readRaplMaxEnergy_uj();
    ec->running = 1;
    pthread_create(&ec->poll_thread, NULL, _rapl_poll_fn, ec);
}

static void stopEnergy(EnergyCounter* ec) {
    ec->running = 0;
    pthread_join(ec->poll_thread, NULL);
}

static double getElapsedEnergy(EnergyCounter ec) {
    return ec.total_energy_j;
}

static void printElapsedEnergy(EnergyCounter ec) {
    printf("Energy: %f J\n", getElapsedEnergy(ec));
}

/* Combined timer + energy measurement
 * Convenience struct for single-run performance + energy measurement.
 */

typedef struct TimerEnergy {
    Timer timer;
    EnergyCounter energy;
} TimerEnergy;

static void startTimerEnergy(TimerEnergy* te) {
    startEnergy(&te->energy);
    startTimer(&te->timer);
}

static void stopTimerEnergy(TimerEnergy* te) {
    stopTimer(&te->timer);
    stopEnergy(&te->energy);
}

static void printTimerEnergy(TimerEnergy te) {
    printf("Elapsed time: %f ms\n", getElapsedTime(te.timer) * 1e3);
    printf("Energy: %f J\n", getElapsedEnergy(te.energy));
}

#endif
