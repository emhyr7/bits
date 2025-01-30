#include <stdio.h>
#include <immintrin.h>
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
typedef unsigned long      Count;
typedef int                Boolean;
typedef long long          Word;

typedef struct {
	Bits64 *pointer;
	Index   index;
} BitLocation;

#define LMASK(x) (~(-1ll << (x)))

static inline BitLocation DoFindBit(Boolean reverse, Boolean clear, Bits64 *p, Bits64 *q) {
	BitLocation result;
	__mmask8 m;
	__m512i v, cmp;
	
	cmp = _mm512_set1_epi64(clear ? -1 : 0);
	if (reverse) p -= 7;
	for (;;) {
		if (p == q) break;

		v = _mm512_load_epi64(p);
		m = _mm512_cmpneq_epu64_mask(v, cmp);
		if (m) break;

		long long inc = sizeof(v) / sizeof(*p);
		if (reverse) inc = -inc;
		p += inc;
	}

	int zcnt;
	if (reverse) zcnt = 7 - _lzcnt_u32((unsigned)m << 24);
	else zcnt = _tzcnt_u32(m);
	result.pointer = p + zcnt;
	if (result.pointer != q) {
		Bits64 x = *result.pointer;
		if (clear) x = ~x;
		(void)_BitScanForward64(&result.index, x);
	}
	return result;
}

inline BitLocation FindSetBit         (Bits64 *p, Bits64 *q) { return DoFindBit(0, 0, p, q); }
inline BitLocation FindClearBit       (Bits64 *p, Bits64 *q) { return DoFindBit(0, 1, p, q); }
inline BitLocation ReverseFindSetBit  (Bits64 *p, Bits64 *q) { return DoFindBit(1, 0, p, q); }
inline BitLocation ReverseFindClearBit(Bits64 *p, Bits64 *q) { return DoFindBit(1, 1, p, q); }

#define WIDTHOF(x) (sizeof(x) * 8)
#define _ffsll(x)  __builtin_ffsll(x)

BitLocation FindClearBits(Size n, Bits64 *p, Bits64 *q) {
	if (n == 1) return FindClearBit(p, q);
	
	BitLocation result;
	Size c, i, j, z;
	Bits64 w;
	
	result.pointer = p;
	result.index = 0;
	c = 0;
	for (; p < q; ++p) {
		w = ~*p;
	retry:
		result.pointer = p;
		i = _ffsll(w);
		if (!i) continue;
		result.index = i - 1;
		w |= ~(-1ll << (i - 1));
		j = _ffsll(~w);
		if (!j) j = WIDTHOF(*p) + 1;
		c += j - i;
		if (c >= n) break;
		w &= -1ll << (j - 1);
		if (!w) continue;
		if (j == WIDTHOF(*p) + 1) {
			do {
				w = *++p;
				if (p >= q) goto finished;
				z = _tzcnt_u64(w);
				c += z;
				if (c >= n) goto finished;
			} while (z == WIDTHOF(*p));
			w |= ~(-1ll << (z - 1));
			w = ~w;
		}
		c = 0;
		goto retry;
	}
finished:
	if (p >= q) result = (BitLocation){0, 0};
	return result;
}

BitLocation correct;

size_t ClockRate;
size_t Start, End;

#define BeginClock() (void)QueryPerformanceCounter(&Start)
#define EndClock()   (void)QueryPerformanceCounter(&End)
#define Elapse()     ((End - Start) * 1000000000 / ClockRate)

#define N 1000000
_Alignas(__m512i) Bits64 p[N];
Bits64 *q = p + N;

int main(int argc, char *argv[]) {
	QueryPerformanceFrequency(&ClockRate);
	memset(p, -1, N * sizeof(Bits64));
	unsigned long long random = N - 1;
	//_rdrand64_step(&random);
	correct.pointer = p + random % N;
	memset(correct.pointer, random, sizeof(Bits64));
	Assert(_BitScanForward64(&correct.index, ~*correct.pointer));
	printf("random %% N: %llu\n", random % N);
	printf("random: %llu\n", random);
	printf("correct.pointer: %p -> %llu\n", correct.pointer, *correct.pointer);
	printf("correct.index: %lu\n", correct.index);
	printf("q: %p\n", q);

	printf("\n\n");

	BitLocation result;

	printf("FindClearBit:\n");
	QueryPerformanceCounter(&Start);
	result = FindClearBit(p, q);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, *result.pointer, result.index);
	
	printf("ReverseFindClearBit:\n");
	QueryPerformanceCounter(&Start);
	result = ReverseFindClearBit(q - 1, p - 1);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, *result.pointer, result.index);
	
	printf("FindClearBits:\n");
	QueryPerformanceCounter(&Start);
	result = FindClearBits(3, p, q);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, result.pointer ? *result.pointer : 0, result.index);
}
