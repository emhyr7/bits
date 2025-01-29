#include <stdio.h>
#include <immintrin.h>
#include <assert.h>
#include <intrin.h>
#include <stdlib.h>
#include <memory.h>

extern int __stdcall QueryPerformanceCounter  (unsigned long long *);
extern int __stdcall QueryPerformanceFrequency(unsigned long long *);

extern void *__stdcall VirtualAlloc(void *, unsigned long long, unsigned, unsigned);

#define Assert(x) do { if (!(x)) __debugbreak(); } while (0)

typedef long long          Bits64;
typedef unsigned long long Size;
typedef unsigned long      Index;

typedef struct {
	Bits64 *pointer;
	Index   index;
} BitLocation;

#define LMASK(x) (~(-1ll << (x)))

static BitLocation FindClearBit2(Bits64 *p, Bits64 *q) {
	BitLocation result;
	Index i;
	__mmask8 m;
	__m512i v, cmp;

	cmp = _mm512_set1_epi64(-1);
	while (p < q) {
		v = _mm512_load_epi64(p);
		m = _mm512_cmplt_epi64_mask(v, cmp);
		if (m) break;
		p += 8;
	}
finish:
	if (_BitScanForward64(&i, m)) {
		p += i;
		result.pointer = p;
		(void)_BitScanForward64(&result.index, ~*p);
		return result;
	} else {
		result = (BitLocation){0, 0};
		return result;
	}
}

size_t ClockRate;
size_t Start, End;

#define BeginClock() (void)QueryPerformanceCounter(&Start)
#define EndClock()   (void)QueryPerformanceCounter(&End)
#define Elapse()     ((End - Start) * 1000000000 / ClockRate)

#define M 10000
#define N (sizeof(__m512i) / sizeof(Bits64) * (M))
_Alignas(__m512i) Bits64 p[N];
Bits64 *q = p + N;

int main(void) {
	QueryPerformanceFrequency(&ClockRate);
	memset(p, -1, N);
	unsigned long long random;
	_rdrand64_step(&random);
	memset(&p[random % M], random, sizeof(__m512i));
	Bits64 *correct_p = &p[random % M];
	BitLocation correct = {correct_p};
	(void)_BitScanForward64(&correct.index, ~*correct_p);

	BitLocation result[1];

	QueryPerformanceCounter(&Start);
	result[0] = FindClearBit2(p, q);
	QueryPerformanceCounter(&End);
	printf("Elapse: %llu\n", Elapse());

	printf("%p & %ld\n", correct.pointer, correct.index);
	for (int i = 0; i < (sizeof(result) / sizeof(result[0])); ++i) {
		printf("%p & %lu\n", result[i].pointer, result[i].index);
	}
}
